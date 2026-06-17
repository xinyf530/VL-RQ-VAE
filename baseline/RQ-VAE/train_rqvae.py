import os
# 限制底层线性代数库的线程数，通常设为 8 或 16 足够快且稳定
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import json
import torch
from tqdm import tqdm

from models.rqvae import RQVAE, QuantizeForwardMode
from utils.data_loader import get_train_val_loaders, get_export_loader

class EarlyStopping:
    """早停机制：监控验证集重建损失"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True # 刷新最佳纪录
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def main():
    # --- 1. 环境与路径配置 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # A100 核心优化：提升 matmul 精度
    torch.set_float32_matmul_precision('high') 
    
    # BASE_DIR = "data/ele"
    BASE_DIR = "data/Toys"
    MODEL_PATH = "rqvae_data/ele"
    os.makedirs(MODEL_PATH, exist_ok=True) # 自动创建结果文件夹
    
    # EMB_PATH = os.path.join(BASE_DIR, "bge_embeddings.pt")
    EMB_PATH = os.path.join(BASE_DIR, "bge_embeddings_v2.pt")
    BEST_MODEL_PATH = os.path.join(MODEL_PATH, "best_rqvae_model.pt")
    
    # --- 2. 数据准备 ---
    # 划分验证集监控收敛
    BATCH_SIZE = 2048
    MAX_EPOCHS = 300
    train_loader, val_loader, num_items = get_train_val_loaders(EMB_PATH, batch_size=BATCH_SIZE)
    export_loader = get_export_loader(EMB_PATH, batch_size=BATCH_SIZE)

    # --- 3. 初始化模型与优化器 ---
    # 采用严格对齐源码的架构和维度
    model = RQVAE(
    input_dim=1024,
    hidden_dims=[512, 256, 128],
    latent_dim=64,
    codebook_size=1024,
    codebook_mode=QuantizeForwardMode.STE,  # 对齐源码配置ROTATION_TRICK STE
    codebook_normalize=False,                           # 对齐源码默认配置
    entropy_weight=0.05,     # 强迫模型把商品打散，像摊大饼一样铺满 1024 个坑 0.1、0.05、0.001、0.005
    restart_threshold=1.0,  # 只要某个簇的使用率 EMA 低于 0.5，立刻被随机输入替换 1.0、0.05、0.1
).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=40)

    print(f"开始训练 (bf16模式 | 严格源码对齐) | 物品数: {num_items} | 显卡: {device}")
    
    # =========================================================
    # 核心修复 1：K-Means 一次性触发机制 (只在 Epoch 0 之前执行一次)
    # =========================================================
    model.train()
    
    # 针对 40 万规模数据，将 KMeans 初始化样本量提升至 200,000
    NUM_INIT_SAMPLES = 20000
    
    print(f"\n[初始化] 准备抓取 {NUM_INIT_SAMPLES} 样本进行 KMeans 冷启动...")
    init_data = []
    for batch in train_loader:
        init_data.append(batch[0])
        if sum(len(b) for b in init_data) >= NUM_INIT_SAMPLES:
            break
            
    # 截取精确的数量并送入 GPU
    init_data = torch.cat(init_data, dim=0)[:NUM_INIT_SAMPLES].to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(init_data) # 触发 quantize.py 中的 _kmeans_init
            
    print("KMeans 初始化完成，正式进入 Epoch 训练！\n")
    # =========================================================

    # --- 4. 训练循环 ---
    for epoch in range(MAX_EPOCHS):
        model.train()
        t_recon, t_vq, t_ent = 0, 0, 0
        
        #pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False, dynamic_ncols=True)
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            # 使用 bf16 自动混合精度
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 解包只接收 3 个参数（去掉了 entropy）
                x_recon, vq_loss, ent_val, _ = model(x, gumbel_t=0.2)
                
                # 对齐源码：先逐样本相加再统一 mean，与 (recon + vq).mean() 结构一致
                recon_loss = ((x_recon - x)**2).sum(axis=-1)   # shape: [B]
                loss = (recon_loss + vq_loss).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_recon += recon_loss.mean().item()
            pure_vq = vq_loss.mean().item() + (0.3 * ent_val.item()) 
            t_vq += pure_vq
            if isinstance(ent_val, torch.Tensor):
                t_ent += ent_val.item()
            
        # --- 验证阶段 (bf16) ---
        model.eval()
        v_recon = 0
        with torch.no_grad():
            for batch in val_loader:
                vx = batch[0].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    vx_recon, _, _, _ = model(vx, gumbel_t=0.2)
                    # 验证集保持与训练集一致的结构
                    v_recon += ((vx_recon - vx)**2).sum(axis=-1).mean().item()
        
        avg_v_recon = v_recon / len(val_loader)
        avg_t_recon = t_recon / len(train_loader)
        avg_t_ent = t_ent / len(train_loader)

        
        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | Train Recon: {avg_t_recon:.5f} | Val Recon: {avg_v_recon:.5f} | VQ Loss: {t_vq/len(train_loader):.5f} | Ent Loss: {avg_t_ent:.5f}")
        
        # 早停与最佳模型保存
        if early_stopping(avg_v_recon):
            print(f"验证集损失刷新纪录: {avg_v_recon:.5f}，保存模型...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
        if early_stopping.early_stop:
            print(f"提前停止：验证损失已连续 {early_stopping.patience} 代未下降。")
            break

    # --- 5. 最终导出：加载最佳权重 ---
    print("\n训练结束，加载最佳模型导出数据...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    all_codes = []
    with torch.no_grad():
        for batch in tqdm(export_loader, desc="导出中"):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, _, _, codes = model(batch[0].to(device), gumbel_t=0.2)
                # 调整形状为 [Batch, num_layers] (若与预期不符可去掉 rearrange)
                all_codes.append(codes.cpu())
    
    all_codes = torch.cat(all_codes, dim=0)
    
    # 文件 1: 映射字典 (.json)
    item2code = {str(i + 1): code.tolist() for i, code in enumerate(all_codes)}
    json_path = os.path.join(MODEL_PATH, "item2code_baseline.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(item2code, f)
    
    # 文件 2: 物理码本 (.pt)
    # 提取所有量化层的 weight
    codebooks = torch.stack([layer.embedding.weight.data.cpu() for layer in model.layers])
    pt_path = os.path.join(MODEL_PATH, "codebooks_baseline.pt")
    torch.save(codebooks, pt_path)
    
    print("任务完成！")
    print(f"   - 映射文件: {json_path}")
    print(f"   - 码本文件: {pt_path}")

if __name__ == "__main__":
    main()