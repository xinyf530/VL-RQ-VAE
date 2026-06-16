

import numpy as np
import scipy.sparse as sp
import torch
import itertools
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD, PCA
import json


def build_svd_cf_embeddings(user_sequences, original_embeddings, num_items, svd_dim=256, pca_target_dim=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[*] 1. 极速构建共现矩阵...")
    rows, cols = [], []
    for seq in tqdm(user_sequences, desc="提取边"):
        unique_items = list(set(seq))
        for i, j in itertools.combinations(unique_items, 2):
            rows.extend([i, j])
            cols.extend([j, i])
            
    data = np.ones(len(rows), dtype=np.float32)
    co_occurrence = sp.coo_matrix((data, (rows, cols)), shape=(num_items, num_items)).tocsr()
    
    # 对频次取 Log，防止超级爆款主导整个矩阵，保留次热物品特征
    co_occurrence.data = np.log1p(co_occurrence.data)


    # ==================================================
    # [替换点] 纯 GPU 极速奇异值分解 (SVD)
    # ==================================================
    print(f"[*] 2. 转移至 GPU 执行极速 Randomized SVD (维度={svd_dim})...")
    coo = co_occurrence.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = torch.Size(coo.shape)
    
    adj_gpu = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    # 🌟 利用 40G 显存直接秒算
    U, S, V = torch.svd_lowrank(adj_gpu, q=svd_dim, niter=2)
    pure_cf_emb = (U * S).cpu().numpy()
    print(f"    -> GPU SVD 计算完成！特征维度: {pure_cf_emb.shape}")

    # ==================================================
    # 恢复原流程：拼接与 PCA
    # ==================================================
    print("[*] 3. 归一化并与原始 BGE 拼接...")
    norm_cf = torch.nn.functional.normalize(torch.FloatTensor(pure_cf_emb), p=2, dim=1)
    norm_bge = torch.nn.functional.normalize(torch.FloatTensor(original_embeddings), p=2, dim=1)
    
    # RQVAE==========================================================
    # 🌟 旋钮二：削弱 SVD 的物理尺度，比如让文本占 75%，SVD 仅占 25%
    alpha = 0.75
    weighted_text = norm_bge * alpha
    weighted_cf = norm_cf * (1.0 - alpha)
    
    concat_emb = torch.cat([weighted_text, weighted_cf], dim=-1).numpy()
    # RQVAE==========================================================
    
    #concat_emb = torch.cat([norm_bge, norm_cf], dim=-1).numpy()
    
    print(f"[*] 4. 执行 PCA 降维回 {pca_target_dim} 维...")
    pca = PCA(n_components=pca_target_dim, random_state=42)
    fused_embeddings = pca.fit_transform(concat_emb)
    
    # 最终的 L2 归一化
    fused_embeddings = torch.nn.functional.normalize(torch.FloatTensor(fused_embeddings), p=2, dim=1)
    
    print("[*] SVD + Concat + PCA 融合完成！")
    return fused_embeddings.cpu().numpy()



def main():
    # ==========================================
    # 1. 加载映射字典，并准备映射关系
    # ==========================================
    print("[*] 加载 id_mapping.json ...")
    with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/id_mapping.json', 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    
    item2id = id_mapping['item2id']
    num_items = len(item2id)
    print(f"[*] 共有 {num_items} 个物品。")

    # ==========================================
    # 2. 读取原始字符串交互序列，并转换为 0-based 整数索引
    # ==========================================
    print("[*] 加载 all_item_seqs.json 并转换序列 ...")
    with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/all_item_seqs.json', 'r', encoding='utf-8') as f:
        all_item_seqs = json.load(f)
        
    user_seqs_0based = []
    for user_raw, seq_raw in all_item_seqs.items():
        # 提取序列并转为 0-based 索引 (即 item2id 中的值减 1)
        mapped_seq = []
        for item in seq_raw:
            if item in item2id:
                mapped_seq.append(item2id[item] - 1) 
        
        if mapped_seq:
            user_seqs_0based.append(mapped_seq)

    # ==========================================
    # 3. 加载原始的 BGE Embeddings
    # ==========================================
    print("[*] 加载原始 BGE Embeddings ...")
    original_embeddings = torch.load('/workspace/user_code/baseline/RQ-VAE/data/Toys/bge_embeddings_text.pt', weights_only=False).cpu().numpy()
    
    # 安全校验：确保 Embedding 数量与物品总数一致
    assert original_embeddings.shape[0] == num_items, \
        f"Embedding行数 ({original_embeddings.shape[0]}) 与 字典物品数 ({num_items}) 不一致！"

    # ==========================================
    # 4. 运行 SVD 纯结构协同融合算法 
    # ==========================================
    # 动态获取原始 BGE 的维度 (比如 1024 或 768)
    original_dim = original_embeddings.shape[1] 
    
    fused_emb_np = build_svd_cf_embeddings(
        user_sequences=user_seqs_0based, 
        original_embeddings=original_embeddings, 
        num_items=num_items, 
        svd_dim=256,                    # 提取 256 维的极锐利图结构特征
        pca_target_dim=original_dim     # PCA 降维回原来的维度，保证无缝接入 RQ-VAE
    )

    # ==========================================
    # 5. 转换为 Tensor 并保存为 .pt
    # ==========================================
    print("[*] 将结果转换为 PyTorch Tensor 并保存...")
    fused_emb_tensor = torch.from_numpy(fused_emb_np).float()
    
    save_path = '/workspace/user_code/baseline/RQ-VAE/data/Toys/bge_embeddings_v2.pt'
    torch.save(fused_emb_tensor, save_path)
    print(f"[*] 成功！融合后的特征已保存至 {save_path}")
    print(f"[*] Tensor 形状: {fused_emb_tensor.shape} (0-based 索引，行0代表item_id 1)")

if __name__ == "__main__":
    main()



"""
import numpy as np
import scipy.sparse as sp
import torch
import itertools
from tqdm import tqdm
from sklearn.decomposition import PCA
import json

# 新函数concat之后1024d
def build_statistical_cf_embeddings(user_sequences, original_embeddings, num_items, 
                                        fusion_mode='concat', alpha=0.5, pca_target_dim=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 启动极速版协同融合，当前设备: {device}，模式: {fusion_mode.upper()}")
    
    # ==================================================
    # 第一阶段：CPU 极速构建共现矩阵 (秒级)
    # ==================================================
    print("[*] 1. 极速构建共现矩阵...")
    item_counts = np.zeros(num_items, dtype=np.float32)
    
    rows = []
    cols = []
    
    # 使用 itertools.combinations 彻底消灭双重 for 循环的低效
    for seq in tqdm(user_sequences, desc="构建共现图"):
        unique_items = list(set(seq))
        # 统计单品出现次数
        for item in unique_items:
            item_counts[item] += 1
            
        # 提取所有两两组合，分别作为 (i, j) 和 (j, i) 存入
        for i, j in itertools.combinations(unique_items, 2):
            rows.extend([i, j])
            cols.extend([j, i])
            
    print(f"[*] 解析到 {len(rows)} 条共现边，正在转换为 CSR 矩阵...")
    # COO 矩阵的核弹级特性：如果有重复的 (row, col) 坐标，它的值会自动累加！
    data = np.ones(len(rows), dtype=np.float32)
    co_occurrence = sp.coo_matrix((data, (rows, cols)), shape=(num_items, num_items))
    co_occurrence = co_occurrence.tocsr() # 转为 CSR 格式加速后续切片

    # ==================================================
    # 第二阶段：GPU 加速归一化与特征聚合 (榨干 40G 显存)
    # ==================================================
    print("[*] 2. 转移至 GPU 计算对称归一化与特征聚合...")
    
    # 提取非零元素的坐标和值，转入 GPU 稀疏张量
    coo = co_occurrence.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = torch.Size(coo.shape)
    
    # 构建 GPU 稀疏矩阵 A
    adj_gpu = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    # 构建 D^(-1/2) 惩罚权重
    item_counts_tensor = torch.FloatTensor(item_counts).to(device)
    item_counts_safe = torch.clamp(item_counts_tensor, min=1.0)
    d_inv_sqrt = torch.pow(item_counts_safe, -0.5)
    
    # 在 GPU 上极速执行 D^(-1/2) * A * D^(-1/2)
    # 直接操作稀疏张量的 values，避免把 40G 显存撑爆
    row_indices = adj_gpu.indices()[0]
    col_indices = adj_gpu.indices()[1]
    norm_values = adj_gpu.values() * d_inv_sqrt[row_indices] * d_inv_sqrt[col_indices]
    
    # 构建带惩罚权重的矩阵
    norm_adj_gpu = torch.sparse_coo_tensor(adj_gpu.indices(), norm_values, shape).coalesce()
    
    # 行归一化 (L1): 计算每行的和，再相除
    row_sums = torch.sparse.sum(norm_adj_gpu, dim=1).to_dense()
    row_sums_safe = torch.clamp(row_sums, min=1e-9)
    final_norm_values = norm_adj_gpu.values() / row_sums_safe[norm_adj_gpu.indices()[0]]
    
    # 最终的聚合权重矩阵 W
    final_adj_gpu = torch.sparse_coo_tensor(norm_adj_gpu.indices(), final_norm_values, shape).coalesce()
    
    print("[*] 3. 执行 GPU 稀疏矩阵乘法 (SpMM) 生成邻居画像...")
    orig_emb_gpu = torch.FloatTensor(original_embeddings).to(device)
    
    # 🌟 核弹级计算：稀疏矩阵 x 稠密特征矩阵
    neighbor_emb_gpu = torch.sparse.mm(final_adj_gpu, orig_emb_gpu)
    
    # 冷启动处理：没有邻居的商品，保留自身特征
    no_neighbor_mask = (item_counts_tensor == 0)
    neighbor_emb_gpu[no_neighbor_mask] = orig_emb_gpu[no_neighbor_mask]

    # ==================================================
    # 第三阶段：特征融合分支 (Add vs Concat)
    # ==================================================
    if fusion_mode == 'add':
        print(f"[*] 4. 在 GPU 上执行加权相加 (alpha={alpha})...")
        # 🌟 修复 Bug：必须先把因为平均池化而严重萎缩的邻居向量“拉长”回长度 1.0！
        norm_neighbor = torch.nn.functional.normalize(neighbor_emb_gpu, p=2, dim=1)

        fused_emb_gpu = alpha * orig_emb_gpu + (1 - alpha) * norm_neighbor
        # 最终 L2 归一化
        fused_emb_gpu = torch.nn.functional.normalize(fused_emb_gpu, p=2, dim=1)
        fused_embeddings = fused_emb_gpu.cpu().numpy()
        
    elif fusion_mode == 'concat':
        print("[*] 4. 在 GPU 上执行 L2 归一化与 Concat 拼接...")
        norm_original = torch.nn.functional.normalize(orig_emb_gpu, p=2, dim=1)
        norm_neighbor = torch.nn.functional.normalize(neighbor_emb_gpu, p=2, dim=1)
        
        concat_emb_gpu = torch.cat([norm_original, norm_neighbor], dim=-1)
        # 传回 CPU 给 scikit-learn 做 PCA
        concat_emb = concat_emb_gpu.cpu().numpy()
        
        print(f"[*] 5. 执行 PCA 降维至 {pca_target_dim} 维...")
        pca = PCA(n_components=pca_target_dim)
        fused_embeddings = pca.fit_transform(concat_emb)
        print(f"[*] PCA 保留的解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # 最终 L2 归一化 (使用 torch 加速一下)
        fused_embeddings = torch.nn.functional.normalize(
            torch.FloatTensor(fused_embeddings).to(device), p=2, dim=1
        ).cpu().numpy()
        
    else:
        raise ValueError("fusion_mode 必须是 'add' 或 'concat'")
        
    print("[*] 融合完成！\n")
    return fused_embeddings


def main():
    # ==========================================
    # 1. 加载映射字典，并准备映射关系
    # ==========================================
    print("[*] 加载 id_mapping.json ...")
    with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/id_mapping.json', 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    
    item2id = id_mapping['item2id']
    num_items = len(item2id)
    print(f"[*] 共有 {num_items} 个物品。")

    # ==========================================
    # 2. 读取原始字符串交互序列，并转换为 0-based 整数索引
    # ==========================================
    print("[*] 加载 all_item_seqs.json 并转换序列 ...")
    with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/all_item_seqs.json', 'r', encoding='utf-8') as f:
        all_item_seqs = json.load(f)
        
    user_seqs_0based = []
    for user_raw, seq_raw in all_item_seqs.items():
        # 提取序列并转为 0-based 索引 (即 item2id 中的值减 1)
        mapped_seq = []
        for item in seq_raw:
            if item in item2id:
                mapped_seq.append(item2id[item] - 1) 
        
        if mapped_seq:
            user_seqs_0based.append(mapped_seq)

    # ==========================================
    # 3. 加载原始的 BGE Embeddings
    # ==========================================
    print("[*] 加载原始 BGE Embeddings ...")
    # 【请根据你的实际文件格式修改这里】
    # 如果你是 numpy 格式：
    # original_embeddings = np.load('bge_emb.npy')
    # 如果你是 torch 的 .pt 格式 (且按照 0-based 排好了序)：
    original_embeddings = torch.load('/workspace/user_code/baseline/RQ-VAE/data/Toys/bge_embeddings_text.pt', weights_only=False).cpu().numpy()
    
    # 安全校验：确保 Embedding 数量与物品总数一致
    assert original_embeddings.shape[0] == num_items, \
        f"Embedding行数 ({original_embeddings.shape[0]}) 与 字典物品数 ({num_items}) 不一致！"

    # ==========================================
    # 4. 运行融合算法 (这里演示 'concat' 模式)
    # ==========================================
    fused_emb_np = build_statistical_cf_embeddings(
        user_sequences=user_seqs_0based, 
        original_embeddings=original_embeddings, 
        num_items=num_items, 
        fusion_mode='concat',      # 如果想用相加模式，改成 'add'
        alpha=0.7, 
        pca_target_dim=original_embeddings.shape[1]  # 降维回 BGE 原本的维度 (如 768)
    )

    # ==========================================
    # 5. 转换为 Tensor 并保存为 .pt
    # ==========================================
    print("[*] 将结果转换为 PyTorch Tensor 并保存...")
    fused_emb_tensor = torch.from_numpy(fused_emb_np).float()
    
    save_path = '/workspace/user_code/baseline/RQ-VAE/data/Toys/bge_embeddings_v1.pt'
    torch.save(fused_emb_tensor, save_path)
    print(f"[*] 成功！融合后的特征已保存至 {save_path}")
    print(f"[*] Tensor 形状: {fused_emb_tensor.shape} (0-based 索引，行0代表item_id 1)")

if __name__ == "__main__":
    main()
"""
