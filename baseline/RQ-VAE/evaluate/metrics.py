import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
import math



def _decode_latents(model, latents, batch_size=2048):
    """辅助函数：将 latent 向量通过 Decoder 还原为 1024 维"""
    device = next(model.parameters()).device
    model.eval()
    recon_embs = []
    dataset = TensorDataset(latents)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            z = batch[0].to(device)
            # 通过解码器重建
            recon = model.decoder(z)
            recon_embs.append(recon.cpu().float())
    return torch.cat(recon_embs, dim=0)

def calc_pareto_frontier(embeddings, all_codes, codebooks, model):
    """
    计算重构误差与平均长度的帕累托前沿点。
    支持变长 SID：按物品实际使用的层数分组，
    截断长度 t 表示「只使用前 t 层码字」的子集物品的平均重构误差。
    """
    num_layers, _, dim = codebooks.shape
    results = []

    for test_len in range(1, num_layers + 1):
        # 只选取实际长度 >= test_len 的物品（它们才有第 test_len 层的码字）
        valid_indices = [i for i, code in enumerate(all_codes) if len(code) >= test_len]
        if not valid_indices:
            continue

        # 用前 test_len 层码字组装 Latent 向量
        recon_latents = torch.zeros((len(valid_indices), dim))
        for out_i, orig_i in enumerate(valid_indices):
            code = all_codes[orig_i]
            for l in range(test_len):
                recon_latents[out_i] += codebooks[l, code[l]]

        # 通过 Decoder 还原到原始维度
        recon_embs = _decode_latents(model, recon_latents)

        # 只与对应物品的原始 Embedding 对比
        orig_embs = embeddings[valid_indices]
        mse = F.mse_loss(recon_embs, orig_embs).item()
        cos_sim = F.cosine_similarity(recon_embs, orig_embs, dim=-1).mean().item()

        results.append({
            'truncate_len': test_len,
            'item_count': len(valid_indices),   # 新增：参与计算的物品数
            'mse': mse,
            'cos_sim': cos_sim
        })

    return results

def calc_length_variance(embeddings, all_codes, codebooks, model):
    """计算不同长度 Bucket 中的平均簇内残差能量"""
    dim = codebooks.shape[2]
    len_to_indices = defaultdict(list)
    recon_latents = torch.zeros((len(all_codes), dim))
    
    for i, code in enumerate(all_codes):
        length = len(code)
        len_to_indices[length].append(i)
        for l in range(length):
            recon_latents[i] += codebooks[l, code[l]]
            
    # 通过 Decoder 批量还原
    recon_embs = _decode_latents(model, recon_latents)
    
    results = []
    for length, indices in sorted(len_to_indices.items()):
        subset_recon = recon_embs[indices]
        subset_orig = embeddings[indices]
        item_mses = F.mse_loss(subset_recon, subset_orig, reduction='none').mean(dim=1)
        results.append({
            'length': length,
            'item_count': len(indices),
            'avg_mse': item_mses.mean().item()
        })
    return results

def calc_collision_rate(all_codes):
    """计算哈希冲突率 (Collision Rate)"""
    code_tuples = [tuple(code) for code in all_codes]
    unique_sids = len(set(code_tuples))
    num_items = len(all_codes)
    collided_items = num_items - unique_sids
    collision_rate = (collided_items / num_items) * 100
    
    return num_items, unique_sids, collided_items, collision_rate

def calc_codebook_utilization(all_codes, num_layers, codebook_size):
    """计算各层码表利用率、死节点与困惑度"""
    results = []
    for layer in range(num_layers):
        layer_tokens = [code[layer] for code in all_codes if len(code) > layer]
        if not layer_tokens:
            continue
            
        layer_counter = Counter(layer_tokens)
        active_nodes = len(layer_counter)
        dead_ratio = (codebook_size - active_nodes) / codebook_size * 100
        
        probs = np.array(list(layer_counter.values())) / len(layer_tokens)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        perplexity = np.exp(entropy)
        
        results.append({
            'layer': layer + 1,
            'active_nodes': active_nodes,
            'dead_ratio': dead_ratio,
            'perplexity': perplexity
        })
    return results

def calc_semantic_preservation(embeddings, all_codes, sample_pairs=100000, return_raw=False):
    """
    计算语义距离保持度（皮尔逊/斯皮尔曼相关系数）。
    支持变长 SID：SID 相似度采用归一化前缀匹配分，
    公式为 prefix_match_len / max(len_a, len_b)，
    长度差异越大惩罚越重，值域为 [0, 1]，连续性更好。
    """
    num_items = len(all_codes)
    idx_A = torch.randint(0, num_items, (sample_pairs,))
    idx_B = torch.randint(0, num_items, (sample_pairs,))

    valid_mask = idx_A != idx_B
    idx_A = idx_A[valid_mask]
    idx_B = idx_B[valid_mask]

    emb_A = embeddings[idx_A]
    emb_B = embeddings[idx_B]
    true_sims = F.cosine_similarity(emb_A, emb_B, dim=-1).numpy()

    sid_sims = []
    for a, b in zip(idx_A.numpy(), idx_B.numpy()):
        code_a = all_codes[a]
        code_b = all_codes[b]

        # 前缀匹配长度（遇到第一个不同即停止）
        match_score = 0
        min_len = min(len(code_a), len(code_b))
        for l in range(min_len):
            if code_a[l] == code_b[l]:
                match_score += 1
            else:
                break

        # 归一化：除以两者中较长的那个，长度差异自动体现为惩罚
        max_len = max(len(code_a), len(code_b))
        sid_sims.append(match_score / max_len)

    sid_sims = np.array(sid_sims)

    pearson_corr, p_value_p = pearsonr(true_sims, sid_sims)
    spearman_corr, p_value_s = spearmanr(true_sims, sid_sims)

    if return_raw:
        return pearson_corr, p_value_p, spearman_corr, p_value_s, true_sims, sid_sims
    return pearson_corr, p_value_p, spearman_corr, p_value_s


def calc_recall_ndcg_id(predictions, ground_truths, k_list=[1, 5, 10]):
    """
    计算推荐系统核心指标:Recall@K 和 NDCG@K
    :param predictions: 二维列表, shape [num_users, max_k], 存放预测的 Top-K 商品 ID
    :param ground_truths: 一维列表, shape [num_users], 存放真实的下一个商品 ID
    :param k_list: 需要计算的 K 值列表，如 [1, 5, 10]
    :return: dict, 包含各个 K 值下的 Recall 和 NDCG
    """
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = 0.0
        metrics[f'NDCG@{k}'] = 0.0

    num_users = len(ground_truths)
    if num_users == 0:
        return metrics

    for preds, target in zip(predictions, ground_truths):
        for k in k_list:
            top_k_preds = preds[:k]
            if target in top_k_preds:
                metrics[f'Recall@{k}'] += 1.0
                
                # 计算排名 (0-indexed)
                rank = top_k_preds.index(target)
                # NDCG 公式: 1 / log2(rank + 1 + 1)  (加1是因为公式里rank是1-indexed)
                metrics[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)

    # 求所有用户的平均值
    for k in k_list:
        metrics[f'Recall@{k}'] /= num_users
        metrics[f'NDCG@{k}'] /= num_users

    return metrics


def calc_recall_ndcg(predictions, ground_truths, k_list=[1, 5, 10]):
    """
    计算推荐系统核心指标: Recall@K 和 NDCG@K (直接对比变长 SID)
    
    :param predictions: 预测的 Top-K SID 列表, shape [num_users, max_k, sid_length] 
                        (内部元素可以是 list, tuple, 或 tensor)
    :param ground_truths: 真实的下一个 SID, shape [num_users, sid_length]
    :param k_list: 需要计算的 K 值列表，如 [1, 5, 10]
    :return: dict, 包含各个 K 值下的 Recall 和 NDCG
    """
    # 初始化 metrics 字典
    metrics = {f'Recall@{k}': 0.0 for k in k_list}
    metrics.update({f'NDCG@{k}': 0.0 for k in k_list})

    num_users = len(ground_truths)
    if num_users == 0:
        return metrics

    for preds, target in zip(predictions, ground_truths):
        # 1. 类型统一化 (极其重要)
        # 如果模型输出或 Target 是 PyTorch/NumPy Tensor，直接用 == 或 in 会报错
        # 所以我们统一将其转化为 Python 原生的 list，方便直接对比内容
        if hasattr(target, 'tolist'):
            target = target.tolist()
        else:
            target = list(target)
            
        # 2. 寻找真实 SID 在预测列表中的排名 (0-indexed)
        hit_rank = -1
        for idx, p in enumerate(preds):
            # 将单个预测结果也转为 list
            p_list = p.tolist() if hasattr(p, 'tolist') else list(p)
            
            # Python 的 list == 会严格比对里面的每一个元素以及长度
            # 完美兼容变长 SID 的比对！
            if p_list == target:
                hit_rank = idx
                break  # 找到最高排名的命中即可跳出
                
        # 3. 如果成功命中，一次性更新所有满足条件的 K 值指标
        if hit_rank != -1:
            for k in k_list:
                if hit_rank < k:  # 只有命中位置在 K 之内才算该 K 指标的命中
                    metrics[f'Recall@{k}'] += 1.0
                    metrics[f'NDCG@{k}'] += 1.0 / math.log2(hit_rank + 2)

    # 4. 求所有用户的平均值
    for k in k_list:
        metrics[f'Recall@{k}'] /= num_users
        metrics[f'NDCG@{k}'] /= num_users

    return metrics


def calc_binned_recall_ndcg_sid(predictions, ground_truths, tensor2bucket, k_list=[1, 5, 10]):
    """
    全自动动态分箱的 Recall 和 NDCG 计算
    """
    # 🌟 动态获取所有箱子的名字，并加上 Overall
    unique_buckets = list(set(tensor2bucket.values()))
    buckets = unique_buckets + ["Overall"]
    
    # 初始化 metrics 和计数器
    metrics = {b: {f'Recall@{k}': 0.0 for k in k_list} for b in buckets}
    for b in buckets:
        metrics[b].update({f'NDCG@{k}': 0.0 for k in k_list})
    counts = {b: 0 for b in buckets}

    num_users = len(ground_truths)
    if num_users == 0:
        return metrics, counts

    for preds, target in zip(predictions, ground_truths):
        target_list = target.tolist() if hasattr(target, 'tolist') else list(target)
        
        # O(1) 查字典获取箱子标签
        target_tuple = tuple(target_list)
        bucket = tensor2bucket.get(target_tuple, "Unknown")
        if bucket not in counts:
            continue

        # 寻找命中排名
        hit_rank = -1
        for idx, p in enumerate(preds):
            p_list = p.tolist() if hasattr(p, 'tolist') else list(p)
            if p_list == target_list:
                hit_rank = idx
                break 

        # 同时更新当前箱子和全局(Overall)指标
        for b in [bucket, "Overall"]:
            counts[b] += 1
            if hit_rank != -1:
                for k in k_list:
                    if hit_rank < k:
                        metrics[b][f'Recall@{k}'] += 1.0
                        metrics[b][f'NDCG@{k}'] += 1.0 / math.log2(hit_rank + 2)

    # 取平均
    for b in buckets:
        if counts[b] > 0:
            for k in k_list:
                metrics[b][f'Recall@{k}'] /= counts[b]
                metrics[b][f'NDCG@{k}'] /= counts[b]

    return metrics, counts

def calc_binned_recall_ndcg(predictions, ground_truths, item2bucket, k_list=[1, 5, 10, 20, 50]):
    """
    全自动动态分箱的 Item-level Recall 和 NDCG 计算
    """
    unique_buckets = list(set(item2bucket.values()))
    buckets = unique_buckets + ["Overall"]
    
    metrics = {b: {f'Recall@{k}': 0.0 for k in k_list} for b in buckets}
    for b in buckets:
        metrics[b].update({f'NDCG@{k}': 0.0 for k in k_list})
    counts = {b: 0 for b in buckets}

    num_users = len(ground_truths)
    if num_users == 0:
        return metrics, counts

    # preds 已经是 item ID 的列表了，target_item 是真实的单个 item ID
    for preds, target_item in zip(predictions, ground_truths):
        bucket = item2bucket.get(target_item, "Unknown")
        if bucket not in counts:
            continue

        # 寻找命中排名
        hit_rank = -1
        for idx, p_item in enumerate(preds):
            if p_item == target_item:
                hit_rank = idx
                break 

        # 更新指标
        for b in [bucket, "Overall"]:
            counts[b] += 1
            if hit_rank != -1:
                for k in k_list:
                    if hit_rank < k:
                        metrics[b][f'Recall@{k}'] += 1.0
                        metrics[b][f'NDCG@{k}'] += 1.0 / math.log2(hit_rank + 2)

    # 取平均
    for b in buckets:
        if counts[b] > 0:
            for k in k_list:
                metrics[b][f'Recall@{k}'] /= counts[b]
                metrics[b][f'NDCG@{k}'] /= counts[b]

    return metrics, counts