import os
import torch
import json
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from dataset.dataset_tiger import TigerDataset, VarlenTigerDataset
from models.model import Tiger
from evaluate.metrics import calc_binned_recall_ndcg


def main():
    # ==========================================
    # 1. Initialize multi-GPU accelerator
    # ==========================================
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 50)
        print(f"启动 TIGER 多卡测试集评估 (GPU 数量: {accelerator.num_processes})")
        print("=" * 50)

    torch.set_float32_matmul_precision('high')

    # 2. Hyperparameters (must match training config)
    MAX_SEQ_LEN = 50
    MAX_TOKEN_LEN = 16
    EVAL_BEAM = 64
    K_LIST = [1, 5, 10, 20, 50]
    MAX_K = max(K_LIST)
    TEST_BATCH_SIZE = 128  # per-GPU batch size for generation

    BASE_DIR = "data/phone"
    #TIGER_DIR = "/workspace/user_code/ED-RQVAE/edrqvae_data/Toys/v5"
    #TIGER_DIR = "/workspace/user_code/baseline/RQ-VAE/rqvae_data/Toys/cf"
    #TIGER_DIR = "/workspace/user_code/baseline/varlen_semantic_ids/result/Toys/varlen2"
    TIGER_DIR = "/workspace/user_code/baseline/RQ-VAE/rqvae_data/phone"

    save_dir = "saves"
    #save_dir = "/workspace/my_folder/luozijian/ED-RQVAE/RQ-VAE/Toys/ed-rqvae"
    #save_dir = "/workspace/my_folder/luozijian/ED-RQVAE/RQ-VAE/Toys/rqvae"
    #save_dir = "/workspace/my_folder/luozijian/ED-RQVAE/RQ-VAE/Toys/varlen"
    
    #best_model_path = os.path.join(save_dir, "tiger_best_var_16.pt")
    #best_model_path = os.path.join(save_dir, "tiger_best_v4_rq.pt")
    best_model_path = os.path.join(save_dir, "tiger_best_phone_2D.pt")
    
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/ed-rqvae/train_log_v4_8.txt"
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/rqvae/train_log_v4.txt"
    #log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/Toys/varlen/train_log_v3_test.txt"
    log_file_path = "/workspace/user_code/baseline/RQ-VAE/logs/phone/train_log_2D.txt"


    # ==========================================
    # 3. Load test dataset
    # ==========================================
    if accelerator.is_main_process:
        print("\n[*] 正在加载测试集 Dataset...")

    test_dataset = VarlenTigerDataset(
        seqs_file=f"{BASE_DIR}/all_item_seqs.json",
        mapping_file=f"{BASE_DIR}/id_mapping.json",
        tiger_code_file=f"{TIGER_DIR}/item2code_final.json",
        #tiger_code_file=f"{TIGER_DIR}/tiger_item2code.json",
        meta_file=f"{TIGER_DIR}/tiger_item2code_meta.json",
        max_seq_len=MAX_SEQ_LEN,
        #max_token_len=MAX_TOKEN_LEN,
        split='test'
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    DYNAMIC_USER_VOCAB_SIZE = len(test_dataset.user2id) + 1

    # ==========================================
    # 4. Initialize model and load checkpoint
    # ==========================================
    if accelerator.is_main_process:
        print("\n[*] 正在初始化 TIGER 模型并加载最佳权重...")

    model = Tiger(
        total_vocab_size=test_dataset.total_vocab_size,
        vocab_sizes=[
            test_dataset.meta[f"vocab_size_layer{i+1}"]
            for i in range(test_dataset.sid_length)
        ],
        user_vocab_size=DYNAMIC_USER_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        #max_token_len=MAX_TOKEN_LEN,
        sid_length=test_dataset.sid_length,
        d_model=256, nhead=4, num_layers=8, dropout=0.15
    )

    # Load best checkpoint
    with accelerator.main_process_first():
            checkpoint = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    if accelerator.is_main_process:
        print(f"[*] 已加载最佳模型 (来自 Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?'):.4f})")

    # Prepare model and dataloader with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)

    # ==========================================
    # 5. Build global SID dictionary (修正：补齐PAD与去重)
    # ==========================================
    if accelerator.is_main_process:
        print("\n[*] 正在加载与构建全局 SID 校验字典...")
        
    all_codes = []
    # 遍历所有存在的商品 ID，让 test_dataset 去完成加 Offset 和 补齐 PAD 的繁琐工作
    for item_id_str in test_dataset.item2code.keys():
        padded_code = test_dataset._get_offset_sid_target(item_id_str)
        all_codes.append(padded_code)
        
    valid_codes_tensor = torch.tensor(all_codes, dtype=torch.long, device=device)
    valid_codes_tensor = torch.unique(valid_codes_tensor, dim=0) # 去重提速
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.build_trie_dfa(valid_codes_tensor)

    # ==========================================
    # 6. Run full test set evaluation
    # ==========================================
    if accelerator.is_main_process:
        print("\n[*] 开始全量测试集 Beam Search 评估...")

    model.eval()
    final_preds_list = []
    # 🌟 新增：追踪用户 ID 以获取真实的 Item ID
    final_users_list = [] 

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test Beam Search]",
                    dynamic_ncols=True, disable=not accelerator.is_main_process)
        for batch in pbar:
            best_paths, _ = accelerator.unwrap_model(model).generate(
                batch=batch,
                n_candidates=256,
                k=EVAL_BEAM,
                temperature=1.0
            )
            
            gathered_paths = accelerator.gather_for_metrics(best_paths)
            # 🌟 新增：收集 User IDs
            gathered_users = accelerator.gather_for_metrics(batch.user_ids)

            if accelerator.is_main_process:
                final_preds_list.extend(gathered_paths.cpu().tolist())
                final_users_list.extend(gathered_users.cpu().tolist())

    # ==========================================
    # 7. Compute metrics (完美 Item ID 展开与分箱)
    # ==========================================
    if accelerator.is_main_process:
        print("\n[*] 正在构建 SID 到 Item ID 的倒排索引...")
        
        # 1. 构建字典：User ID -> 真实的 Target Item ID
        user2target_item = {int(u): str(i) for u, _, i in test_dataset.samples}
        
        # 🌟 新增：统计所有 Item 在训练集中的全局流行度 (用于簇内排序)
        print("\n[*] 正在统计全局流行度...")
        item_popularity = {}
        for raw_item_seq in test_dataset.all_seqs.values():
            train_seq = raw_item_seq[:-2] # 只用训练集数据
            for raw_item in train_seq:
                item_int_str = str(test_dataset.item2id[raw_item])
                item_popularity[item_int_str] = item_popularity.get(item_int_str, 0) + 1
        
        # 2. 构建字典：SID (Tuple) -> [Item_A, Item_B, ...] (处理碰撞)
        sid2items = {}
        for item_int_str in test_dataset.item2code.keys():
            padded_code = tuple(test_dataset._get_offset_sid_target(item_int_str))
            if padded_code not in sid2items:
                sid2items[padded_code] = []
            sid2items[padded_code].append(item_int_str)

        # 🌟 核心黑科技：对倒排字典里的每个簇，按流行度降序预排序！
        for padded_code in sid2items:
            sid2items[padded_code].sort(
                key=lambda x: item_popularity.get(x, 0), 
                reverse=True
            )

        print("\n[*] 开始执行 Item 级别的 Beam Search 展开 (带流行度优化)...")
        final_preds = []
        final_gts = []

        for i in range(len(final_users_list)):
            user_id = final_users_list[i]
            gt_item = user2target_item[user_id]
            
            pred_sids = final_preds_list[i]
            pred_items = []
            seen_sids = set()
            
            for beam_idx in range(EVAL_BEAM):
                if len(pred_items) >= MAX_K:
                    break
                pred_sid = tuple(pred_sids[beam_idx])
                
                if pred_sid in seen_sids:
                    continue
                seen_sids.add(pred_sid)
                
                # 此时拿到的 mapped_items 已经是把热门商品排在前面的了！
                mapped_items = sid2items.get(pred_sid, [])
                
                for item in mapped_items:
                    if item not in pred_items:
                        pred_items.append(item)
                        
            final_preds.append(pred_items[:MAX_K])
            final_gts.append(gt_item)

        print("\n[*] 正在预构建 分箱映射字典...")
        train_counts = {}
        for raw_item_seq in test_dataset.all_seqs.values():
            train_seq = raw_item_seq[:-2] 
            for raw_item in train_seq:
                item_int_str = str(test_dataset.item2id[raw_item])
                train_counts[item_int_str] = train_counts.get(item_int_str, 0) + 1

        # 🌟 修改：这次我们直接把 Item ID 映射到 Bucket，不再需要 Tensor 映射
        item2bucket = {}
        for item_int_str in test_dataset.item2code.keys():
            freq = train_counts.get(item_int_str, 0)
            if freq < 6:
                item2bucket[item_int_str] = "Cold (<=5)"
            else:
                item2bucket[item_int_str] = "Hot (>=6)"
                
        print(f"[✓] 字典构建完成！成功映射 {len(item2bucket)} 个商品的分箱。")

        # 传入新的 metrics 函数
        test_metrics, counts = calc_binned_recall_ndcg(final_preds, final_gts, item2bucket, k_list=K_LIST)

        # ==========================================
        # 8. Print and log results (震撼的分箱打印)
        # ==========================================
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_msg = (
            f"\n{'='*65}\n"
            f"  TIGER 变长测试集 全量与分箱评估结果 ({current_time})\n"
            f"{'='*65}\n"
            f"模型来源: {best_model_path} (Epoch {checkpoint.get('epoch', '?')})\n"
            f"Beam Size: {EVAL_BEAM} | n_candidates: 256\n"
            f"{'-'*65}\n"
        )
        
        # 格式化各个箱子的打印输出
        for bucket in ["Overall", "Hot (>=6)", "Cold (<=5)"]:
            if counts.get(bucket, 0) == 0:
                continue
                
            r_str = ", ".join([f"{test_metrics[bucket][f'Recall@{k}']:.4f}" for k in K_LIST])
            n_str = ", ".join([f"{test_metrics[bucket][f'NDCG@{k}']:.4f}" for k in K_LIST])
            
            log_msg += (
                f"[{bucket:^16}] 测试样本数: {counts[bucket]:<6}\n"
                f"   Recall@[1,5,10,20,50]: [{r_str}]\n"
                f"     NDCG@[1,5,10,20,50]: [{n_str}]\n"
            )
            if bucket == "Overall":
                log_msg += f"{'-'*65}\n"
                
        log_msg += f"{'='*65}"

        print(log_msg)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
        print(f"\n[✓] 结果已追加写入 {log_file_path}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()