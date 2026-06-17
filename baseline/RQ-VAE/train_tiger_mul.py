import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import json
import datetime
from accelerate import Accelerator

from dataset.dataset_tiger import TigerDataset, VarlenTigerDataset
from models.model import Tiger
from evaluate.metrics import calc_recall_ndcg


class InverseSquareRootScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return self.base_lrs
        scale_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
        return [base_lr * scale_factor for base_lr in self.base_lrs]

def main():
    # ==========================================
    # 1. 初始化多卡加速器
    # ==========================================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("="*50)
        print(f"启动 TIGER 分布式训练 (GPU 数量: {accelerator.num_processes})")
        print("="*50)

    torch.set_float32_matmul_precision('high')

    # 2. 核心超参数设置
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.0005
    EPOCHS = 100       
    MAX_SEQ_LEN = 50
    MAX_TOKEN_LEN = 16
    PATIENCE = 5
    WARMUP_STEPS = 1770 # 2080 235 1770

    # BASE_DIR = "data/phone"
    #TIGER_DIR = "/workspace/user_code/ED-RQVAE/edrqvae_data/Toys/v5"
    #TIGER_DIR = "/workspace/user_code/baseline/RQ-VAE/rqvae_data/Toys/cf"
    #TIGER_DIR = "/workspace/user_code/baseline/varlen_semantic_ids/result/Toys/varlen2"
    # TIGER_DIR = "/workspace/user_code/baseline/RQ-VAE/rqvae_data/phone"
    BASE_DIR = "/data1/xinyuefeng/VL-RQ-VAE/baseline/RQ-VAE/data/Toys/AmazonReviews2014/Toys/processed"
    TIGER_DIR = "/data1/xinyuefeng/VL-RQ-VAE/baseline/RQ-VAE/rqvae_data/ele"

    if accelerator.is_main_process:
        print("\n[*] 正在加载与构建 Dataset...")
    
    # 3. 初始化 Dataset 和 DataLoader
    train_dataset = VarlenTigerDataset(
        seqs_file=f"{BASE_DIR}/all_item_seqs.json",
        mapping_file=f"{BASE_DIR}/id_mapping.json",
        tiger_code_file=f"{TIGER_DIR}/item2code_final.json",
        #tiger_code_file=f"{TIGER_DIR}/tiger_item2code.json",
        # meta_file=f"{TIGER_DIR}/tiger_item2code_meta.json",
        meta_file=f"{TIGER_DIR}/item2code_final_meta.json",
        max_seq_len=MAX_SEQ_LEN,
        #max_token_len=MAX_TOKEN_LEN,
        split='train' 
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=train_dataset.collate_fn
    )

    val_dataset = VarlenTigerDataset(
        seqs_file=f"{BASE_DIR}/all_item_seqs.json",
        mapping_file=f"{BASE_DIR}/id_mapping.json",
        tiger_code_file=f"{TIGER_DIR}/item2code_final.json",
        #tiger_code_file=f"{TIGER_DIR}/tiger_item2code.json",
        # meta_file=f"{TIGER_DIR}/tiger_item2code_meta.json",
        meta_file=f"{TIGER_DIR}/item2code_final_meta.json",
        max_seq_len=MAX_SEQ_LEN,
        #max_token_len=MAX_TOKEN_LEN,
        split='val'  
    )
    val_loss_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=val_dataset.collate_fn
    )
    
    VAL_BATCH_SIZE = 64
    val_gen_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, 
        collate_fn=val_dataset.collate_fn
    )

    DYNAMIC_USER_VOCAB_SIZE = len(train_dataset.user2id) + 1
    
    if accelerator.is_main_process:
        print("\n[*] 正在初始化 TIGER 模型...")
        
    # 4. 初始化 TIGER 模型
    model = Tiger(
        total_vocab_size=train_dataset.total_vocab_size,
        vocab_sizes=[
            train_dataset.meta[f"vocab_size_layer{i+1}"]
            for i in range(train_dataset.sid_length)
        ],
        user_vocab_size=DYNAMIC_USER_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        #max_token_len=MAX_TOKEN_LEN,
        sid_length=train_dataset.sid_length,
        d_model=256, nhead=4, num_layers=8, dropout=0.15
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=WARMUP_STEPS
    )

    model, optimizer, train_loader, val_loss_loader, val_gen_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loss_loader, val_gen_loader
) 

    if accelerator.is_main_process:
        print("\n[*] 正在加载与构建全局 SID 校验字典...")

    # 1. 加上偏移量，并使用 PAD_TOKEN_ID 强行补齐为规整矩阵
    all_codes = []
    # 直接调用 Dataset 里已经写得天衣无缝的方法，它能完美处理一切定长、变长、UID的排列组合！
    for item_id_str in train_dataset.item2code.keys():
        padded_code = train_dataset._get_offset_sid_target(item_id_str)
        all_codes.append(padded_code)
        
    # 2. 转换为 Tensor 并执行全局去重！(极大提升后续 Beam Search 校验速度)
    valid_codes_tensor = torch.tensor(all_codes, dtype=torch.long, device=device)
    valid_codes_tensor = torch.unique(valid_codes_tensor, dim=0)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.build_trie_dfa(valid_codes_tensor)

    best_val_loss = float('inf')
    early_stop_counter = 0
    save_dir = "saves"
    best_model_path = os.path.join(save_dir, "tiger_best_phone_2D.pt")
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/ed-rqvae/test.txt"
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/rqvae/train_log_v4.txt"
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/varlen/train_log_v3_test.txt"
    log_file_path = "/data1/xinyuefeng/VL-RQ-VAE/baseline/RQ-VAE/logs/train_log_2D.txt"
    
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    def print_and_log(msg):
        """仅在主卡执行打印并追加写入 txt 文件"""
        if accelerator.is_main_process:
            print(msg)
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    # 记录训练起始时间
    current_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_and_log(f"\n{'='*50}\n多卡分布式训练正式开始 ({current_start_time})\n{'='*50}")

    # 5. 正式训练循环与早停逻辑
    for epoch in range(EPOCHS):
        # --- 5.1 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        
        # 进度条只在主卡显示
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", 
                    dynamic_ncols=True, disable=not accelerator.is_main_process)
                    
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            loss, _ = model(batch)
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            lr_scheduler.step()
            total_train_loss += loss.item()

            
            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                                "LR": f"{current_lr:.6f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 5.2 极速计算验证集 Loss ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loss_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val Loss]",
                            dynamic_ncols=True, disable=not accelerator.is_main_process)
            for batch in val_pbar:
                loss, _ = model(batch)
                gathered_loss = accelerator.gather(loss)
                total_val_loss += gathered_loss.mean().item()
                val_pbar.set_postfix({"Val Loss": f"{gathered_loss.mean().item():.4f}"})
        
        avg_val_loss = total_val_loss / len(val_loss_loader)
        accelerator.wait_for_everyone()

        # --- 5.3 早停判断与条件抽样生成 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': best_val_loss,
                }, best_model_path)

            # --- 抽样验证 (多卡并行生成并汇总) ---
            local_preds = []
            local_gts = []
            NUM_SAMPLE_BATCHES = 3
            EVAL_BEAM = 64          
            K_LIST = [1, 5, 10, 20, 50]
            MAX_K = max(K_LIST)
            
            with torch.no_grad():
                gen_pbar = tqdm(val_gen_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Beam Search]",
                                total=NUM_SAMPLE_BATCHES, dynamic_ncols=True,
                                disable=not accelerator.is_main_process)
                for step, batch in enumerate(gen_pbar):
                    if step >= NUM_SAMPLE_BATCHES: break
                    
                    best_paths, _ = accelerator.unwrap_model(model).generate(
                        batch=batch,
                        #valid_codes_tensor=valid_codes_tensor,
                        n_candidates=64, 
                        k=EVAL_BEAM
                    )
                    
                    local_preds.append(best_paths)
                    local_gts.append(batch.sem_ids_fut)
            
            if len(local_preds) > 0:
                gathered_paths = accelerator.gather_for_metrics(torch.cat(local_preds, dim=0))
                gathered_gts = accelerator.gather_for_metrics(torch.cat(local_gts, dim=0))
                
                if accelerator.is_main_process:
                    paths_list = gathered_paths.tolist()
                    gts_list = gathered_gts.tolist()
                    
                    final_preds = []
                    final_gts = []
                    
                    for i in range(len(gts_list)):
                        gt_sid = gts_list[i] # 这是一个包含 PAD 的 SID List，比如 [107, 898, 2048, 2048]
                        
                        pred_sids = []
                        for beam_idx in range(EVAL_BEAM):
                            if len(pred_sids) >= MAX_K: break
                            pred_sid = paths_list[i][beam_idx]
                            
                            # 基于纯语义 SID 去重：
                            # Beam Search 可能会生成微小概率差异但结果完全相同的路径
                            if pred_sid not in pred_sids:
                                pred_sids.append(pred_sid)
                        
                        final_preds.append(pred_sids)
                        final_gts.append(gt_sid)

                    # 直接将二维的预测 SID 列表和一维的真实 SID 列表丢进评估函数
                    sample_metrics = calc_recall_ndcg(final_preds, final_gts, k_list=K_LIST)

                    # 使用写入日志和打印一体化函数
                    r_str = ", ".join([f"{sample_metrics[f'Recall@{k}']:.4f}" for k in K_LIST])
                    n_str = ", ".join([f"{sample_metrics[f'NDCG@{k}']:.4f}" for k in K_LIST])
                    
                    log_msg = (
                        f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (新低) | 已保存 tiger_best.pt\n"
                        f"   ↳ Recall@[1,5,10,20,50]: [{r_str}]\n"
                        f"   ↳   NDCG@[1,5,10,20,50]: [{n_str}]"
                    )
                    print_and_log(log_msg)

        else:
            early_stop_counter += 1
            log_msg = f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (未降) | 早停计数: {early_stop_counter}/{PATIENCE}"
            print_and_log(log_msg)
            
            if early_stop_counter >= PATIENCE:
                stop_msg = f"连续 {PATIENCE} 个 Epoch 未提升，触发早停！最佳模型在: {best_model_path}"
                print_and_log(stop_msg)
                break
            
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()