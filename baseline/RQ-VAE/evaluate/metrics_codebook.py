import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
import random
from datetime import datetime
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. 动态环境路径配置 (彻底解决导包报错)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from models.rqvae import RQVAE
except ImportError:
    print("⚠️ 警告: 无法导入 models.rqvae，维度三 (帕累托前沿) 将被跳过。")
    RQVAE = None


# ==========================================
# 1. 工具函数：加载商品类目、名称与真实热度
# ==========================================
def load_item_info(mapping_path, meta_path, interactions_path=None):
    """
    解析类目、完整标题，并基于序列文件精确计算每个商品的真实交互热度
    """
    print("[准备数据] 正在解析类目、商品名称与热度信息...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    item2id = mapping.get('item2id', {})
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    item2category = {}
    item2title = {}
    item2pop = {} 
    valid_cat_count = 0
    
    # 1.1 基础映射与元数据提取
    for asin, item_id in item2id.items():
        idx = int(item_id) - 1  # 映射到 0-based Tensor 索引
        text = meta.get(asin, "")
        
        # 提取 Category
        cat_match = re.search(r"Main Category:\s*(.*?)\.", text)
        if cat_match:
            item2category[idx] = cat_match.group(1).strip()
            valid_cat_count += 1
        else:
            item2category[idx] = "Unknown"
            
        # 提取完整 Title
        title_match = re.search(r"Title:\s*(.*?)\.", text)
        item2title[idx] = title_match.group(1).strip() if title_match else "Unknown Item"
        
        # 初始化所有商品热度为 0
        item2pop[idx] = 0

    # 1.2 解析序列文件，统计全局真实热度
    if interactions_path and os.path.exists(interactions_path):
        print(f"  -> 正在统计商品交互热度 (来源: {os.path.basename(interactions_path)})...")
        with open(interactions_path, 'r', encoding='utf-8') as f:
            all_seqs = json.load(f)
            
        # 遍历所有用户的历史交互序列
        for user_id, seq in all_seqs.items():
            for asin in seq:
                if asin in item2id:
                    idx = int(item2id[asin]) - 1
                    item2pop[idx] += 1
                    
        total_interactions = sum(item2pop.values())
        print(f"  -> 热度统计完成！共计 {total_interactions} 次真实交互记录。")
    else:
        print("  -> ⚠️ 未提供或未找到交互序列文件，热度统计将被跳过。")
        
    return item2category, item2title, item2pop


# ==========================================
# 2. 核心类：ED-RQVAE 专属质量评估器
# ==========================================
class SIDQualityEvaluator:
    def __init__(self, embeddings, all_codes, item2category=None, item2title=None, item2pop=None, codebooks=None, model=None):
        self.embeddings = embeddings  
        self.all_codes = all_codes
        self.num_items = len(all_codes)
        self.item2category = item2category
        self.item2title = item2title
        self.item2pop = item2pop
        self.codebooks = codebooks    
        self.model = model            
        
        # 聚类字典: SID tuple -> item_indices
        self.cluster_dict = defaultdict(list)
        for idx, code in enumerate(self.all_codes):
            self.cluster_dict[tuple(code)].append(idx)

    # ---------------------------------------------------------
    # 维度一：资源利用与宏观健康度
    # ---------------------------------------------------------
    def calc_dead_code_ratio(self, num_layers=3, codebook_size=1024):
        results = []
        for layer in range(num_layers):
            layer_tokens = [code[layer] for code in self.all_codes if len(code) > layer]
            if not layer_tokens: continue
            active_nodes = len(set(layer_tokens))
            dead_ratio = (codebook_size - active_nodes) / codebook_size * 100
            results.append({'layer': layer + 1, 'dead_ratio': dead_ratio})
        return results

    def calc_max_cluster_size(self):
        max_size = max([len(indices) for indices in self.cluster_dict.values()] + [0])
        bh_ratio = (max_size / self.num_items) * 100 if self.num_items > 0 else 0
        return max_size, bh_ratio

    # ---------------------------------------------------------
    # 维度二：语义保真度与碰撞质量
    # ---------------------------------------------------------
    def calc_global_intra_variance_and_purity(self):
        total_variance, total_purity, valid_clusters = 0.0, 0.0, 0
        for code_tuple, indices in self.cluster_dict.items():
            K = len(indices)
            if K > 1:
                cluster_embs = self.embeddings[indices]
                centroid = cluster_embs.mean(dim=0, keepdim=True)
                total_variance += ((cluster_embs - centroid) ** 2).sum(dim=1).mean().item()
                if self.item2category:
                    categories = [self.item2category[idx] for idx in indices]
                    total_purity += (Counter(categories).most_common(1)[0][1] / K) * 100
                valid_clusters += 1
        return (total_variance / valid_clusters if valid_clusters else 0.0, 
                total_purity / valid_clusters if valid_clusters else 0.0, 
                valid_clusters)

    def calc_knn_prefix_recall(self, k=10, prefix_len=1, sample_size=10000):
        sample_size = min(sample_size, self.num_items)
        sample_indices = torch.randint(0, self.num_items, (sample_size,))
        query_embs = self.embeddings[sample_indices]
        
        sim_scores = query_embs @ self.embeddings.T 
        _, topk_indices = torch.topk(sim_scores, k=k+1, dim=1)
        
        hit_count, total_neighbors = 0, 0
        for i, query_idx in enumerate(sample_indices.tolist()):
            query_code = self.all_codes[query_idx]
            if len(query_code) < prefix_len: continue
            query_prefix = tuple(query_code[:prefix_len])
            
            neighbor_indices = [idx for idx in topk_indices[i].tolist() if idx != query_idx][:k]
            for n_idx in neighbor_indices:
                neighbor_code = self.all_codes[n_idx]
                if len(neighbor_code) >= prefix_len and tuple(neighbor_code[:prefix_len]) == query_prefix:
                    hit_count += 1
                total_neighbors += 1
        return (hit_count / total_neighbors) * 100 if total_neighbors > 0 else 0.0

    # 🚀 重点升级：Top-10 黑洞簇全景解剖 (支持类目多样性抽样与热度计算)
    def analyze_topk_crowded_clusters(self, top_k=10):
        sorted_clusters = sorted(self.cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)
        reports = []
        for rank, (code_tuple, indices) in enumerate(sorted_clusters[:top_k]):
            K = len(indices)
            if K <= 1: break
            
            # 1. 计算方差
            cluster_embs = self.embeddings[indices]
            centroid = cluster_embs.mean(dim=0, keepdim=True)
            variance = ((cluster_embs - centroid) ** 2).sum(dim=1).mean().item()
            
            # 2. 计算纯度
            purity_str = "N/A"
            if self.item2category:
                categories = [self.item2category[idx] for idx in indices]
                top1_cat, top1_count = Counter(categories).most_common(1)[0]
                purity_str = f"{(top1_count / K) * 100:.1f}% (主类: {top1_cat})"
                
            # 3. 计算该簇的平均热度
            avg_pop = "未知"
            if self.item2pop and sum(self.item2pop.values()) > 0:
                avg_pop = sum(self.item2pop[idx] for idx in indices) / K
                avg_pop = f"{avg_pop:.1f} 次交互/商品"
                
            # 4. 多样性抽样 (确保不同类目的卧底被揪出来)
            sample_titles = []
            if self.item2title and self.item2category:
                cat_to_indices = defaultdict(list)
                for idx in indices: 
                    cat_to_indices[self.item2category[idx]].append(idx)
                
                diverse_sample_indices = []
                # 第一轮：每个类目各抽一个
                for cat, idx_list in cat_to_indices.items():
                    diverse_sample_indices.append(random.choice(idx_list))
                    if len(diverse_sample_indices) >= 5: break
                
                # 第二轮：不足5个则随机补齐
                if len(diverse_sample_indices) < 5 and len(indices) > len(diverse_sample_indices):
                    remaining_indices = list(set(indices) - set(diverse_sample_indices))
                    needed = min(5 - len(diverse_sample_indices), len(remaining_indices))
                    diverse_sample_indices.extend(random.sample(remaining_indices, needed))
                
                for idx in diverse_sample_indices:
                    cat = self.item2category[idx]
                    title = self.item2title[idx]
                    pop_info = f"[热度:{self.item2pop[idx]}]" if self.item2pop and sum(self.item2pop.values()) > 0 else ""
                    sample_titles.append(f"[{cat}]{pop_info} {title}")
                
            reports.append({
                'rank': rank+1, 'code': code_tuple, 'size': K, 
                'variance': variance, 'purity': purity_str, 'avg_pop': avg_pop,
                'sample_titles': sample_titles
            })
        return reports

    # ---------------------------------------------------------
    # 维度三：编码效率
    # ---------------------------------------------------------
    def calc_pareto_mse(self, max_len=3):
        if self.codebooks is None or self.model is None: return None
        results = []
        device = next(self.model.parameters()).device
        self.model.eval()
        for test_len in range(1, max_len + 1):
            valid_indices = [i for i, c in enumerate(self.all_codes) if len(c) >= test_len]
            if not valid_indices: continue
            
            dim = self.codebooks.shape[2]
            recon_latents = torch.zeros((len(valid_indices), dim))
            for out_i, orig_i in enumerate(valid_indices):
                code = self.all_codes[orig_i]
                for l in range(test_len):
                    recon_latents[out_i] += self.codebooks[l, code[l]]
            
            recon_embs = []
            loader = DataLoader(TensorDataset(recon_latents), batch_size=2048)
            with torch.no_grad():
                for batch in loader:
                    recon = self.model.decoder(batch[0].to(device))
                    recon_embs.append(recon.cpu())
            recon_embs = torch.cat(recon_embs, dim=0)
            
            orig_embs = self.embeddings[valid_indices].cpu()
            results.append({'len': test_len, 'mse': F.mse_loss(recon_embs, orig_embs).item()})
        return results

    # ---------------------------------------------------------
    # 报告生成与导出
    # ---------------------------------------------------------
    def generate_report_string(self):
        report = []
        report.append("="*95)
        report.append(f"📊 语义 ID (SID) 质量诊断报告 - Top-10 簇热度全景剖析 (ED-RQVAE)")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*95)
        
        dead_ratios = self.calc_dead_code_ratio()
        max_size, bh_ratio = self.calc_max_cluster_size()
        global_var, global_purity, valid_clusters = self.calc_global_intra_variance_and_purity()
        knn_recall = self.calc_knn_prefix_recall()
        
        report.append("\n▶ 1. 宏观统计概览")
        for res in dead_ratios:
            report.append(f"   - [Layer {res['layer']}] 码表死节点率: {res['dead_ratio']:.2f}%")
        report.append(f"   - 全局平均簇内方差: {global_var:.6f} | 全局类目纯度: {global_purity:.2f}%")
        report.append(f"   - 真实近邻前缀召回率(L1): {knn_recall:.2f}%")
        report.append(f"   - 最大簇容量: {max_size} 个商品 (占全局 {bh_ratio:.4f}%)")
        
        report.append("\n▶ 2. 🎯 [深度解剖] Top-10 极度拥挤簇 (查明是热门聚集还是冷门坍缩):")
        topk_reports = self.analyze_topk_crowded_clusters(top_k=10)
        if not topk_reports:
            report.append("      ✅ 当前模型无碰撞")
        for rep in topk_reports:
            alert = "🚨[高方差坍缩]" if rep['variance'] > 0.05 else "✅[低方差聚合]"
            report.append(f"\n   🔴 Top {rep['rank']} 簇 {rep['code']} | 容量: {rep['size']} 个商品")
            report.append(f"      ↳ 簇内残差能量: {rep['variance']:.6f} {alert}")
            report.append(f"      ↳ 真实业务纯度: {rep['purity']}")
            if rep['avg_pop'] != "未知":
                report.append(f"      ↳ 簇内平均热度: {rep['avg_pop']} (诊断核心：冷门长尾 or 热门爆款)")
            
            if rep.get('sample_titles'):
                report.append(f"      ↳ 多样性商品抽样 (含热度与全名):")
                for i, title in enumerate(rep['sample_titles']):
                    report.append(f"          {i+1}. {title}")
                    
        mse_results = self.calc_pareto_mse()
        if mse_results:
            report.append("\n▶ 3. 编码效率 (Coding Efficiency)")
            for res in mse_results:
                report.append(f"   3.1 截断长度 {res['len']} Token 时的重构 MSE: {res['mse']:.6f}")
                
        report.append("\n" + "="*95 + "\n")
        return "\n".join(report)

    def save_report(self, save_path):
        report_str = self.generate_report_string()
        print(report_str)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"\n✅ 诊断报告已成功保存至: {os.path.abspath(save_path)}")


# ==========================================
# ⚡️ 极速执行入口
# ==========================================
if __name__ == "__main__":
    # ================= 路径配置区 =================
    BASE_DIR = "../data/Toys"
    BGE_EMB_PATH = f"{BASE_DIR}/bge_embeddings.pt"
    ID_MAPPING_PATH = f"{BASE_DIR}/id_mapping.json"
    METADATA_PATH = f"{BASE_DIR}/metadata.sentence.json"
    
    # 🚀 你的交互序列文件 (用于计算商品真实热度)
    INTERACTIONS_FILE = f"{BASE_DIR}/all_item_seqs.json"
    
    # 你的模型与生成结果
    ITEM2CODE_PATH = "../rqvae_data/Toys/item2code_baseline.json"
    CODEBOOK_PATH = "../rqvae_data/Toys/codebooks_baseline.pt"  
    MODEL_WEIGHTS = "../rqvae_data/Toys/best_rqvae_model.pt"    
    
    # 报告导出路径
    REPORT_SAVE_PATH = "evaluate/Toys/evaluation_report_Top10_Full.txt"
    # ==============================================
    
    print("[1/3] 加载归一化的 BGE Embeddings...")
    bge_embeddings = torch.load(BGE_EMB_PATH, map_location="cpu")
    
    print("[2/3] 解析类目信息、商品 Title 与真实交互热度...")
    item2category, item2title, item2pop = load_item_info(ID_MAPPING_PATH, METADATA_PATH, INTERACTIONS_FILE)
    
    print("[3/3] 加载 SID 分配字典...")
    with open(ITEM2CODE_PATH, "r") as f:
        item2code_dict = json.load(f)
    
    all_codes = [[] for _ in range(len(item2code_dict))]
    for item_str, code in item2code_dict.items():
        all_codes[int(item_str) - 1] = code
        
    print("[4/4] 尝试加载模型以计算重构误差...")
    codebooks, model = None, None
    if RQVAE is not None and os.path.exists(CODEBOOK_PATH) and os.path.exists(MODEL_WEIGHTS):
        try:
            codebooks = torch.load(CODEBOOK_PATH, map_location="cpu")
            model = RQVAE(
                input_dim=1024,
                hidden_dims=[512, 256, 128],
                latent_dim=64,
                codebook_size=1024
            )
            model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
            print("  -> 模型加载成功！")
        except Exception as e:
            print(f"  -> 模型加载失败，跳过维度三。原因: {e}")
    
    # 启动全景评估
    print("\n🚀 正在执行多维评估，请稍候 (包含 KNN、方差、热度统计与多样性抽样)...")
    evaluator = SIDQualityEvaluator(
        embeddings=bge_embeddings, 
        all_codes=all_codes, 
        item2category=item2category,
        item2title=item2title,
        item2pop=item2pop,
        codebooks=codebooks,
        model=model
    )
    evaluator.save_report(REPORT_SAVE_PATH)