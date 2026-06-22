import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm

# 导入你写好的模型结构
from models.ed_rqvae import ED_RQVAE, QuantizeForwardMode

def load_embeddings(emb_path):
    print("Loading original dense embeddings...")
    return torch.load(emb_path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    BASE_DIR = "/workspace/user_code/baseline/RQ-VAE/data/Toys"
    MODEL_PATH = "edrqvae_data/Toys/v2"
    EMB_PATH = os.path.join(BASE_DIR, "bge_embeddings.pt")
    BEST_MODEL_PATH = os.path.join(MODEL_PATH, "best_ed_rqvae_model.pt")
    RAW_CODE_PATH = os.path.join(MODEL_PATH, "item2code_raw.json")
    FINAL_CODE_PATH = os.path.join(MODEL_PATH, "item2code_final.json")
    
    # 1. 加载原始变长 ID
    with open(RAW_CODE_PATH, 'r') as f:
        item2code_raw = json.load(f)
        
    embeddings = load_embeddings(EMB_PATH).to(device)
    
    # 2. 碰撞检测
    code2items = defaultdict(list)
    for item_id_str, code in item2code_raw.items():
        code_tuple = tuple(code)
        code2items[code_tuple].append(int(item_id_str))
        
    collision_groups = {code: items for code, items in code2items.items() if len(items) > 1}
    unique_groups = {code: items for code, items in code2items.items() if len(items) == 1}
    
    print(f"Total items: {len(item2code_raw)}")
    print(f"Unique IDs before expansion: {len(unique_groups)}")
    print(f"Collision groups to expand: {len(collision_groups)}")
    
    # 3. 加载训练好的模型
    model = ED_RQVAE(
        input_dim=1024,
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        codebook_size=1024,
        num_layers=3,
        codebook_mode=QuantizeForwardMode.STE,  
    ).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    # 手动关闭所有层的 KMeans 冷启动标志！
    for layer in model.layers:
        layer.kmeans_initted = True

    # 我们将反复使用最后一层 (Layer 3) 来解决残余碰撞
    last_layer = model.layers[-1]
    
    item2code_final = {}
    
    # 先把没有碰撞的直接保存
    for code, items in unique_groups.items():
        item2code_final[str(items[0])] = list(code)
        
    # 4. 动态深度扩展 (Dynamic Depth Expansion)
    print("Resolving collisions dynamically...")
    with torch.no_grad():
        for base_code, item_ids in tqdm(collision_groups.items(), desc="Expanding"):
            indices = [iid - 1 for iid in item_ids]
            x = embeddings[indices]
                
            # 获取初始特征 (这步是确定的，因为 Encoder 只有矩阵乘法，没有离散化 Argmin)
            res = model.encode(x)
                
            # --- 核心修复开始：绝不重新预测，直接拿确定性的 base_code 取 Embedding！ ---
            for i, token_id in enumerate(base_code):
                layer = model.layers[i]
                # base_code[i] (即 token_id) 就是这批碰撞商品在这层共有的确切 Token ID
                # 纯 GPU 级操作：直接构造全一样的 exact_ids，确保扣除的残差 100% 准确！
                exact_ids = torch.full((len(item_ids),), token_id, dtype=torch.long, device=device)
                
                emb = layer.get_item_embeddings(exact_ids)
                res = res - emb
            # --- 核心修复结束 ---
                
            # 第二步：在残余的 res 上不断自回归量化，只解开“恶性碰撞”，坦然接受“克隆体”！
            current_codes = [list(base_code) for _ in item_ids]
            max_extra_depth = 3
            
            previous_unique_count = 0
            
            for depth in range(max_extra_depth):
                # 如果当前碰撞组里的商品已经全部分开了，立刻退出
                if len(set(tuple(c) for c in current_codes)) == len(current_codes):
                    break
                    
                # 核心理念守护：克隆体检测！
                # 如果往下挖了一层，发现 unique 数量毫无变化，说明剩下的全是一模一样的“克隆体”
                current_unique_count = len(set(tuple(c) for c in current_codes))
                if current_unique_count == previous_unique_count and depth > 0:
                    # 我们绝不加毫无意义的 Unique ID！直接退出，让克隆体共享 SID！
                    break 
                previous_unique_count = current_unique_count
                import torch.nn.functional as F
                # 纯粹的语义量化
                codebook = last_layer.out_proj(last_layer.embedding.weight.data)
                res_normed = F.normalize(res, dim=-1)
                codebook_normed = F.normalize(codebook, dim=-1)
                cos_dist = -(res_normed @ codebook_normed.T)
                new_ids = cos_dist.argmin(dim=-1)

                emb = last_layer.get_item_embeddings(new_ids)
                res = res - emb 
                
                for idx, new_token in enumerate(new_ids.tolist()):
                    current_codes[idx].append(new_token)
            
            # 记录最终的纯语义 IDs
            for iid, final_code in zip(item_ids, current_codes):
                item2code_final[str(iid)] = final_code

    # 5. 保存结果
    with open(FINAL_CODE_PATH, 'w', encoding='utf-8') as f:
        json.dump(item2code_final, f)
        
    print(f"\n扩展完成！100% 纯语义 ID 已保存至: {FINAL_CODE_PATH}")
    
    # 打印一下扩展后的平均长度
    total_len = sum(len(code) for code in item2code_final.values())
    print(f"扩展后的全局词表平均长度: {total_len / len(item2code_final):.3f}")

if __name__ == "__main__":
    main()