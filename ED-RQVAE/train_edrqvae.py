import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import json
import torch
from tqdm import tqdm

from models.ed_rqvae import ED_RQVAE, QuantizeForwardMode
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
            return True 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high') 
    
    BASE_DIR = "/workspace/user_code/baseline/RQ-VAE/data/Toys"
    MODEL_PATH = "edrqvae_data/Toys/v4"
    os.makedirs(MODEL_PATH, exist_ok=True) 
    
    EMB_PATH = os.path.join(BASE_DIR, "bge_embeddings.pt")
    BEST_MODEL_PATH = os.path.join(MODEL_PATH, "best_ed_rqvae_model.pt")
    
    BATCH_SIZE = 2048
    MAX_EPOCHS = 300
    
    SPLIT_INTERVAL = 5            # 每隔 5 个 Epoch 触发一次启发式分裂
    TARGET_LENGTH_BIAS = -1.0      # 控制长度分布的唯一锚点！(-1.0 大概是均长 2.16) -1.5:2.38 -0.5:1.91
    DEAD_THRESHOLD = 1.0          # 死节点判定阈值
    SPLIT_GAMMA = 0.01            # 分裂噪声的基础缩放率
    ENTROPY_WEIGHT = 0.1

    NEIGHBOR_LOSS_START_EPOCH = 10   # Strategy B: only introduce after reconstruction converges
    NEIGHBOR_LOSS_RAMP_EPOCHS = 30    # Ramp up slowly over 50 epochs
    NEIGHBOR_LOSS_MAX_WEIGHT = 0.1   # Strategy B: 10x smaller weight to protect reconstruction


    train_loader, val_loader, num_items = get_train_val_loaders(EMB_PATH, batch_size=BATCH_SIZE)
    export_loader = get_export_loader(EMB_PATH, batch_size=BATCH_SIZE)

    # 初始化 ED_RQVAE
    model = ED_RQVAE(
        input_dim=1024,
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        codebook_size=1024,
        num_layers=3,
        codebook_mode=QuantizeForwardMode.STE,  
        codebook_normalize=False,                               
        entropy_weight=ENTROPY_WEIGHT,     
        ema_decay=0.99,
        target_length_bias=TARGET_LENGTH_BIAS,
        use_hierarchical_init=True
    ).to(device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=40)

    print(f"开始 ED-RQVAE 训练 | 物品数: {num_items} | 显卡: {device}")
    
    # =========================================================
    # K-Means 一次性冷启动
    # =========================================================
    model.train()
    NUM_INIT_SAMPLES = 200000
    print(f"\n[初始化] 准备抓取 {NUM_INIT_SAMPLES} 样本进行 KMeans 冷启动...")
    init_data = []
    for batch in train_loader:
        init_data.append(batch[0])
        if sum(len(b) for b in init_data) >= NUM_INIT_SAMPLES:
            break
            
    init_data = torch.cat(init_data, dim=0)[:NUM_INIT_SAMPLES].to(device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(init_data) 
            
    print("KMeans 初始化完成！\n")

    # =========================================================
    # 核心循环
    # =========================================================
    for epoch in range(MAX_EPOCHS):
        model.train()

        t_recon, t_vq, t_ent, t_neighbor = 0, 0, 0, 0
        total_gen_length = 0.0

        # Progressive weight schedule for neighborhood consistency loss
        if epoch < NEIGHBOR_LOSS_START_EPOCH:
            lambda_neighbor = 0.0
        else:
            progress = min(1.0, (epoch - NEIGHBOR_LOSS_START_EPOCH) / NEIGHBOR_LOSS_RAMP_EPOCHS)
            lambda_neighbor = NEIGHBOR_LOSS_MAX_WEIGHT * progress
        
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # ED-RQVAE 返回 5 个参数
                x_recon, vq_loss, ent_val, sem_ids, gen_lens, z_encoder, x_original = model(x, gumbel_t=0.2)
                
                recon_loss = ((x_recon - x)**2).sum(axis=-1).mean()
                entropy_loss = -ent_val * ENTROPY_WEIGHT

                # Neighborhood consistency loss using original space distance + encoder gradient
                if lambda_neighbor > 0:
                    neighbor_loss = model.compute_neighbor_loss(x_original, z_encoder, sem_ids, gen_lens)
                else:
                    neighbor_loss = torch.tensor(0.0, device=device)

                # Enhanced loss: Recon + VQ + Entropy + Neighborhood Consistency
                loss = recon_loss + vq_loss + entropy_loss + lambda_neighbor * neighbor_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_recon += recon_loss.item()
            t_vq += vq_loss.item()
            total_gen_length += gen_lens.mean().item()
            if isinstance(ent_val, torch.Tensor):
                t_ent += ent_val.item()
            if isinstance(neighbor_loss, torch.Tensor):
                t_neighbor += neighbor_loss.item()
                
        # 核心机制：周期性触发启发式分裂与动量清洗
        if (epoch + 1) % SPLIT_INTERVAL == 0:
            print(f"\n[Epoch {epoch+1}] 触发启发式分裂与 EMA 动量清洗...")
            split_count = model.trigger_heuristic_splitting(dead_threshold=DEAD_THRESHOLD, gamma=SPLIT_GAMMA)
            if split_count == 0:
                print(" -> 当前无高方差死节点，跳过分裂。")
            
        # --- 验证阶段 (bf16) ---
        model.eval()
        v_recon = 0
        with torch.no_grad():
            for batch in val_loader:
                vx = batch[0].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    vx_recon, _, _, _, _, _, _ = model(vx, gumbel_t=0.2)
                    v_recon += ((vx_recon - vx)**2).sum(axis=-1).mean().item()
        
        avg_v_recon = v_recon / len(val_loader)
        avg_t_recon = t_recon / len(train_loader)
        avg_t_len = total_gen_length / len(train_loader)
        avg_t_neighbor = t_neighbor / len(train_loader)

        
        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]
        
        # 增加平均生成长度的监控
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | 均长: {avg_t_len:.2f} | Train Recon: {avg_t_recon:.5f} | Val Recon: {avg_v_recon:.5f} | VQ: {t_vq/len(train_loader):.4f} | λ_N: {lambda_neighbor:.4f} | N_Loss: {avg_t_neighbor:.4f}")
        
        if early_stopping(avg_v_recon):
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
        if early_stopping.early_stop:
            print(f"提前停止：验证损失已连续 {early_stopping.patience} 代未下降。")
            break

    # =========================================================
    # 最终变长特征导出 (直接导出截断后的结果)
    # =========================================================
    print("\n训练结束，加载最佳模型导出变长数据...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    item2code_raw = {}
    item2length_raw = {}
    
    global_item_id = 1
    with torch.no_grad():
        for batch in tqdm(export_loader, desc="导出变长 SIDs"):
            x_batch = batch[0].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 推理模式下，模型会进行硬截断
                _, _, _, codes, gen_lens, _, _ = model(x_batch, gumbel_t=0.2)
                
                # 遍历 Batch 中的每一个商品，执行真实截断
                for i in range(x_batch.size(0)):
                    length = int(gen_lens[i].item())
                    # 安全保护：长度至少为 1
                    length = max(1, min(length, model.num_layers))
                    
                    # 真正的变长！如果 length=1，这里只会切出 [72]
                    varlen_code = codes[i, :length].tolist()
                    
                    item_id_str = str(global_item_id)
                    item2code_raw[item_id_str] = varlen_code
                    item2length_raw[item_id_str] = length
                    
                    global_item_id += 1
    
    json_code_path = os.path.join(MODEL_PATH, "item2code_raw.json")
    json_len_path = os.path.join(MODEL_PATH, "item2length_raw.json")
    
    with open(json_code_path, 'w', encoding='utf-8') as f:
        json.dump(item2code_raw, f)
    with open(json_len_path, 'w', encoding='utf-8') as f:
        json.dump(item2length_raw, f)
        
    print(f"   - 原始变长 SID 字典: {json_code_path}")
    print(f"   - 原始长度字典: {json_len_path}")

if __name__ == "__main__":
    main()