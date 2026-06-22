import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, NamedTuple
from enum import Enum
from einops import rearrange
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class _KmeansOutput(NamedTuple):
    centroids: torch.Tensor
    assignment: torch.Tensor


class _Kmeans:
    def __init__(self, k: int, max_iters: int = None, stop_threshold: float = 1e-10):
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x: torch.Tensor) -> None:
        B, _ = x.shape
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx, :]
        self.assignment = None

    def _update_centroids(self, x: torch.Tensor) -> None:
        # 1. 极速距离计算 (保持上一版)
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        c_sq = (self.centroids ** 2).sum(dim=-1).unsqueeze(0)
        xc = x @ self.centroids.T
        squared_pw_dist = x_sq + c_sq - 2 * xc
        
        # 使用 argmin 比 min().indices 更快
        centroid_idx = squared_pw_dist.argmin(dim=1)
        self.assignment = centroid_idx
        
        # ==========================================
        # 2. 【核心提速区】消除 1024 次 for 循环，使用 One-Hot 并行矩阵乘法
        # ==========================================
        # 将一维索引转为 One-Hot 矩阵，shape: [200000, 1024]
        one_hot = F.one_hot(centroid_idx, num_classes=self.k).to(x.dtype)
        
        # 统计每个聚类中心分到了多少个样本，shape: [1024, 1]
        cluster_counts = one_hot.sum(dim=0, keepdim=True).T
        
        # 并行计算每个聚类的特征总和！shape: [1024, 1024]
        # 解析: [1024, 200000] @ [200000, 1024] = [1024, 1024]
        cluster_sums = one_hot.T @ x 
        
        # 找出分配数量为 0 的空簇
        empty_clusters = (cluster_counts.squeeze() == 0)
        num_empty = empty_clusters.sum().item()
        
        # 计算新的聚类中心 (除以数量，clamp防止除以0)
        new_centroids = cluster_sums / cluster_counts.clamp(min=1e-8)
        
        # 如果有空簇，极其高效地从原始数据中随机抽取填补
        if num_empty > 0:
            random_idx = torch.randint(0, x.size(0), (num_empty,), device=x.device)
            new_centroids[empty_clusters] = x[random_idx]
            
        # QuantizeForwardMode.ROTATION_TRICK用：因为我们是在做球面(L2)量化，KMeans 的重心必须重新投影回球面上！
        #new_centroids = F.normalize(new_centroids, p=2, dim=-1)
        
        self.centroids = new_centroids

    def run(self, x: torch.Tensor) -> _KmeansOutput:
        self._init_centroids(x)
        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroids.clone()
            self._update_centroids(x)
            if torch.norm(self.centroids - old_c, dim=1).max() < self.stop_threshold:
                break
            i += 1
        return _KmeansOutput(centroids=self.centroids, assignment=self.assignment)


def _kmeans_init_(tensor: torch.Tensor, x: torch.Tensor) -> None:
    """Initialize codebook embedding weights with KMeans centroids (in-place)."""
    assert tensor.dim() == 2 and x.dim() == 2
    with torch.no_grad():
        k, _ = tensor.shape
        # 强制最多只跑 50 次迭代，防止死循环
        out = _Kmeans(k=k, max_iters=50).run(x) 
        tensor.data.copy_(out.centroids)

def _hierarchical_kmeans_init_(tensor: torch.Tensor, x: torch.Tensor) -> None:
    """
    Hierarchical KMeans initialization for codebook embeddings (in-place).
    
    Strategy: First cluster data into sqrt(k) coarse groups, then sub-cluster
    each group into sqrt(k) fine clusters. This produces k centroids that
    naturally respect the hierarchical tree structure of RQ-VAE, ensuring
    L1 codes capture coarse semantic regions and deeper codes refine them.
    """
    assert tensor.dim() == 2 and x.dim() == 2
    with torch.no_grad():
        k, dim = tensor.shape

        # Step 1: Determine coarse/fine split
        n_coarse = int(np.ceil(np.sqrt(k)))
        n_fine = int(np.ceil(k / n_coarse))

        # Step 2: Coarse-level clustering
        coarse_km = _Kmeans(k=n_coarse, max_iters=50)
        coarse_out = coarse_km.run(x)

        # Step 3: Sub-cluster each coarse group
        all_centroids = []
        for c_idx in range(n_coarse):
            # Get samples assigned to this coarse cluster
            mask = (coarse_out.assignment == c_idx)
            sub_x = x[mask]

            if len(sub_x) == 0:
                # Empty cluster: generate random centroids near coarse centroid
                noise = torch.randn(n_fine, dim, device=x.device) * 0.01
                all_centroids.append(coarse_out.centroids[c_idx].unsqueeze(0) + noise)
                continue

            if len(sub_x) <= n_fine:
                # Too few samples: use all samples + pad with noisy copies
                pad_count = n_fine - len(sub_x)
                if pad_count > 0:
                    noise = torch.randn(pad_count, dim, device=x.device) * 0.01
                    padded = sub_x[torch.randint(0, len(sub_x), (pad_count,))] + noise
                    all_centroids.append(torch.cat([sub_x, padded], dim=0))
                else:
                    all_centroids.append(sub_x)
                continue

            # Normal case: run fine-level KMeans within this coarse cluster
            fine_km = _Kmeans(k=n_fine, max_iters=30)
            fine_out = fine_km.run(sub_x)
            all_centroids.append(fine_out.centroids)

        # Step 4: Concatenate and trim to exactly k centroids
        all_centroids = torch.cat(all_centroids, dim=0)[:k]

        # Safety: if we got fewer than k (shouldn't happen), pad with random samples
        if all_centroids.shape[0] < k:
            pad_idx = torch.randint(0, x.shape[0], (k - all_centroids.shape[0],))
            all_centroids = torch.cat([all_centroids, x[pad_idx]], dim=0)
            
        tensor.data.copy_(all_centroids)




def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return l2norm(x, dim=self.dim, eps=self.eps)

def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample


def efficient_rotation_trick_transform(u, q, e):
    """
    u: 归一化后的 query  (x / ||x||)
    q: 归一化后的 quantized emb  (emb / ||emb||)
    e: 原始输入 x
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()

class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight=1.0):
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value):
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss

class NeighborhoodConsistencyLoss(nn.Module):
    """
    Neighborhood Consistency Regularization Loss (V3 - Original Space Alignment).
    
    Key improvement: Uses original input space L2 distance for pair similarity
    (aligned with evaluation metric), while maintaining gradient flow through
    encoder output via a differentiable proxy.
    
    Loss = -Pearson(L2_sim_original_space, NPS_min_score)
         + alpha * alignment_loss(encoder_neighbors, original_neighbors)
    """
    def __init__(self, num_pairs=2048):
        super().__init__()
        self.num_pairs = num_pairs
    
    def _compute_lcp_nps_min(self, sem_ids_a: torch.Tensor, sem_ids_b: torch.Tensor,
                            lens_a: torch.Tensor, lens_b: torch.Tensor) -> torch.Tensor:
        """
        Compute NPS-min = LCP(a,b) / min(len(a), len(b)) for each pair.
        All operations are vectorized on GPU.
        """
        match_mask = (sem_ids_a == sem_ids_b)
        cumulative_match = match_mask.float().cumprod(dim=1)
        lcp = cumulative_match.sum(dim=1)
        min_len = torch.min(lens_a, lens_b).clamp(min=1.0)
        nps_min = lcp / min_len
        return nps_min
    
    def forward(self, original_input: torch.Tensor, encoder_output: torch.Tensor,
                sem_ids: torch.Tensor, gen_lens: torch.Tensor):
        """
        Args:
            original_input: [B, D_input] raw input embeddings (1024-dim, no gradient needed)
            encoder_output: [B, D_latent] encoder output (64-dim, has gradient for backprop)
            sem_ids: [B, num_layers] full semantic IDs from all layers
            gen_lens: [B] actual generated lengths for each sample
        Returns:
            loss: scalar, combined loss for neighborhood consistency
        """
        B = original_input.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=original_input.device)
        
        n_pairs = min(self.num_pairs, B * (B - 1) // 2)
        n_hard = n_pairs // 2
        n_rand = n_pairs - n_hard
        
        # ---- Hard pair mining: use ORIGINAL space L2 distance (aligned with eval) ----
        n_anchors = min(n_hard, B)
        anchor_idx = torch.randperm(B, device=original_input.device)[:n_anchors]
        anchor_embs = original_input[anchor_idx]
        
        l2_dists = torch.cdist(anchor_embs, original_input, p=2)
        
        for i, a_idx in enumerate(anchor_idx):
            l2_dists[i, a_idx] = float('inf')
        
        _, nn_idx = l2_dists.min(dim=1)
        
        hard_idx_a = anchor_idx[:n_hard]
        hard_idx_b = nn_idx[:n_hard]
        
        # ---- Random pairs ----
        rand_idx_a = torch.randint(0, B, (n_rand,), device=original_input.device)
        rand_idx_b = torch.randint(0, B, (n_rand,), device=original_input.device)
        same_mask = (rand_idx_a == rand_idx_b)
        rand_idx_b[same_mask] = (rand_idx_b[same_mask] + 1) % B
        
        idx_a = torch.cat([hard_idx_a, rand_idx_a], dim=0)
        idx_b = torch.cat([hard_idx_b, rand_idx_b], dim=0)
        
        # ---- L2 similarity in ORIGINAL space (aligned with eval metric) ----
        orig_a = original_input[idx_a]
        orig_b = original_input[idx_b]
        orig_l2_dist = torch.norm(orig_a - orig_b, p=2, dim=-1)
        orig_l2_sim = 1.0 / (1.0 + orig_l2_dist)  # Same transform as eval script
        
        # ---- Compute continuous NPS-min matching score ----
        nps_min = self._compute_lcp_nps_min(
            sem_ids[idx_a], sem_ids[idx_b],
            gen_lens[idx_a], gen_lens[idx_b]
        )
        
        # ---- Pearson correlation (using original space distance) ----
        sim_mean = orig_l2_sim.mean()
        nps_mean = nps_min.mean()
        
        sim_centered = orig_l2_sim - sim_mean
        nps_centered = nps_min - nps_mean
        
        cov = (sim_centered * nps_centered).mean()
        std_sim = sim_centered.std().clamp(min=1e-6)
        std_nps = nps_centered.std().clamp(min=1e-6)
        
        pearson = cov / (std_sim * std_nps)
        
        # ---- Differentiable proxy: encourage encoder to preserve original-space neighborhoods ----
        # For hard pairs (original-space nearest neighbors), minimize encoder-space distance
        # This makes encoder topology-preserving, so L1 assignments respect original-space structure
        enc_a = encoder_output[hard_idx_a]
        enc_b = encoder_output[hard_idx_b]
        # Pull original-space neighbors closer in encoder space
        enc_dist_hard = ((enc_a - enc_b) ** 2).sum(dim=-1)
        
        # For random pairs that are far in original space, push apart in encoder space
        enc_rand_a = encoder_output[rand_idx_a]
        enc_rand_b = encoder_output[rand_idx_b]
        orig_rand_dist = torch.norm(original_input[rand_idx_a] - original_input[rand_idx_b], p=2, dim=-1)
        
        # Contrastive: pull close pairs, push far pairs
        # Use original-space distance as soft weight
        orig_hard_dist = torch.norm(orig_a[:n_hard] - orig_b[:n_hard], p=2, dim=-1)
        max_dist = orig_rand_dist.max().clamp(min=1e-6)
        
        # Normalized pull loss for hard pairs (neighbors should be close in encoder space)
        pull_loss = enc_dist_hard.mean()
        
        # Normalized push loss for random far pairs (dissimilar items should be far in encoder space)
        margin = 2.0
        enc_dist_rand = ((enc_rand_a - enc_rand_b) ** 2).sum(dim=-1)
        push_loss = F.relu(margin - enc_dist_rand).mean()
        
        # Combined: Pearson alignment + contrastive topology preservation
        # alpha controls the strength of the contrastive proxy
        alpha = 0.1
        contrastive_loss = pull_loss + push_loss
        
        return -pearson + alpha * contrastive_loss



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim, dropout=0.0, normalize=False):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim] + hidden_dims + [out_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.SiLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x):
        assert x.shape[-1] == self.input_dim, \
            f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)



class ED_Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        codebook_normalize: bool = False,
        sim_vq: bool = False,
        do_kmeans_init: bool = True,
        entropy_weight: float = 0.1,
        ema_decay: float = 0.99,  # 统一 EMA 衰减率
        use_hierarchical_init: bool = False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.entropy_weight = entropy_weight
        self.ema_decay = ema_decay
        self.use_hierarchical_init = use_hierarchical_init


        # 核心 1：双重状态追踪寄存器 (Hit & Energy)
        # 记录每个节点的 EMA 命中次数 (N^t)
        self.register_buffer("cluster_usage", torch.ones(n_embed) * 10.0)
        # 记录每个节点的 EMA 残差能量 (E^t)
        self.register_buffer("cluster_energy", torch.zeros(n_embed))

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )
        
        self.quantize_loss = QuantizeLoss(commitment_weight=commitment_weight)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad()
    def _kmeans_init(self, x):
        if self.use_hierarchical_init:
            print(f"[Quantizer] Running Hierarchical KMeans init on GPU (n_embed={self.n_embed})")
            _hierarchical_kmeans_init_(self.embedding.weight, x.float())
        else:
            print(f"[Quantizer] Running KMeans init on GPU (n_embed={self.n_embed})")
            _kmeans_init_(self.embedding.weight, x.float())
        
        self.kmeans_initted = True
    
    def get_item_embeddings(self, item_ids):
        return self.out_proj(self.embedding(item_ids))
    
    def forward(self, x, temperature=0.2):
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)
        
        codebook = self.out_proj(self.embedding.weight)

        # 计算距离
        if self.distance_mode == QuantizeDistance.L2:
            # x shape: [Batch, 64] -> x_sq shape: [Batch, 1]
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            # codebook shape: [1024, 64] -> c_sq shape: [1, 1024]
            c_sq = (codebook ** 2).sum(dim=-1).unsqueeze(0)
            # 矩阵乘法: [Batch, 64] @ [64, 1024] -> [Batch, 1024]
            dist = x_sq + c_sq - 2 * (x @ codebook.T)

        elif self.distance_mode == QuantizeDistance.COSINE:
            x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            c_norm = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dist = -(x_norm @ c_norm.T)

        else:
            raise Exception(f"Unsupported Quantize distance mode: {self.distance_mode}")
        
        _, ids = (dist.detach()).min(axis=1)

        # 核心 2：动态更新命中与残差能量 EMA
        if self.training:
            with torch.no_grad():
                # 1. 更新命中次数 N^t
                counts = torch.bincount(ids, minlength=self.n_embed).float()
                self.cluster_usage.mul_(self.ema_decay).add_(counts, alpha=(1 - self.ema_decay))

                # 2. 获取量化后的离散向量 (用于计算残差)
                emb_detached = self.get_item_embeddings(ids).detach()

                # 3. 计算每个样本当前的残差能量 ||r_t||^2
                sample_energy = ((x.detach() - emb_detached) ** 2).sum(dim=-1)

                # 4. 按类簇累加残差能量
                batch_cluster_energy = torch.zeros(self.n_embed, device=x.device)
                batch_cluster_energy.scatter_add_(0, ids, sample_energy)

                # 5. 更新簇残差能量 E^t
                self.cluster_energy.mul_(self.ema_decay).add_(batch_cluster_energy, alpha=(1 - self.ema_decay))

        # 计算高熵损失 (Entropy Loss)
        entropy_val = torch.tensor(0.0, device=x.device)
        if self.training and self.entropy_weight > 0:
            prob = F.softmax(-dist / temperature, dim=-1)
            avg_prob = prob.mean(dim=0)
            entropy_val = -torch.sum(avg_prob * torch.log(avg_prob + 1e-8))
            # 删除了混淆视听的 entropy_penalty 计算
            
        # 核心前向传播 (Gumbel/STE/Rotation)
        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(-dist, temperature=temperature, device=x.device)
                emb = weights @ codebook
                emb_out = emb
                
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x
                )
                emb_out = emb_out * (
                    torch.norm(emb, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                ).detach()
                
            else:
                raise Exception(f"Unsupported mode: {self.forward_mode}")
                
            vq_loss = self.quantize_loss(query=x, value=emb)

            return emb_out, vq_loss, entropy_val, ids
            
        else:
            emb_out = self.get_item_embeddings(ids)
            vq_loss = self.quantize_loss(query=x, value=emb_out)
            
            return emb_out, vq_loss, entropy_val, ids
    
    # 核心 3：启发式分裂与清洗 (独立于 Forward 之外)
    @torch.no_grad()
    def heuristic_split(self, dead_threshold=1.0, gamma=0.01):
        """
        每隔 K 个 Epoch 调用的启发式分裂函数。
        寻找死节点，并用最高方差(拥挤)节点的能量将其劈开。
        返回: 执行了分裂的节点对数量
        """
        # 1. 找出低于死亡阈值的死节点 D
        dead_mask = self.cluster_usage < dead_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        num_dead = len(dead_indices)

        if num_dead == 0:
            return 0 # 无死节点，无需分裂
        
        # 2. 找出方差(能量)最高的拥挤节点 C
        # 排除掉死节点本身
        valid_energy = self.cluster_energy.clone()
        valid_energy[dead_mask] = -1.0

        # 确保我们要找的拥挤节点数量不会超过可用节点的上限
        num_split = min(num_dead, (valid_energy > 0).sum().item())
        if num_split == 0:
            return 0
        
        dead_indices = dead_indices[:num_split]
        _, crowded_indices = torch.topk(valid_energy, num_split)

        # 3. 对称扰动与 EMA 清洗
        for d, c in zip(dead_indices, crowded_indices):
            base_vec = self.embedding.weight.data[c].clone()

            # 动态噪声缩放：用该拥挤簇的残差能量标准差来缩放 gamma
            # --- 修正点开始 ---
            # 获取该簇的平均能量(即方差)，方能代表真实的物理分布宽度 (Spread)
            # 添加 clamp 防止除以 0 (虽然 EMA 保底了 10.0，但安全第一)
            variance = self.cluster_energy[c] / self.cluster_usage[c].clamp(min=1.0)
            energy_scale = torch.sqrt(variance.clamp(min=1e-6))
            # --- 修正点结束 ---
            #energy_scale = torch.sqrt(self.cluster_energy[c].clamp(min=1e-6))
            noise = torch.randn_like(base_vec) * energy_scale * gamma

            # 空间重置：对称劈开 (确保 bfloat16 兼容)
            self.embedding.weight.data[d] = (base_vec - noise).to(self.embedding.weight.dtype)
            self.embedding.weight.data[c] = (base_vec + noise).to(self.embedding.weight.dtype)

            # EMA 动量清洗
            half_usage = self.cluster_usage[c] / 2.0
            self.cluster_usage[d] = half_usage
            self.cluster_usage[c] = half_usage

            self.cluster_energy[d] = 0.0
            self.cluster_energy[c] = 0.0

        return num_split


# ED-RQVAE 创新组件：能量与密度感知门控 (Energy-Density Halting Gate)
class HaltingGate(nn.Module):
    def __init__(self, hidden_dim=32, target_length_bias=-1.0):
        super().__init__()
        # 输入维度为 2：[残差能量 (Energy), 簇命中密度 (Density)]
        self.net = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 它确保了每个 Batch 内的 Logits 呈标准正态分布 N(0,1)
        self.batch_balancer = nn.BatchNorm1d(1, affine=False)
        # 学习率解耦的锚点：
        # -0.0 -> 约 50% 样本截断
        # -1.0 -> 约 27% 样本截断
        # -2.0 -> 约 12% 样本截断
        self.register_buffer("bias", torch.tensor([target_length_bias]))
    
    def forward(self, energy, density):
        log_density = torch.log(density.clamp(min=1e-6))
        x = torch.cat([energy, log_density], dim=-1)
        
        raw_logits = self.net(x)

        # 训练期间，强行做 Batch 维度的负载均衡！
        # 防止所有样本一起坍缩到 L1
        if self.training and raw_logits.size(0) > 1:
            balanced_logits = self.batch_balancer(raw_logits)
        else:
            balanced_logits = raw_logits

        # 最终输出 = N(0, 1) + 目标截断偏置
        return balanced_logits + self.bias


# ED-RQVAE 主干网络
class ED_RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = None,
        latent_dim: int = 64,
        codebook_size: int = 1024,
        num_layers: int = 3,
        commitment_weight: float = 0.25,
        codebook_normalize: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        entropy_weight: float = 0.1,
        ema_decay: float = 0.99,
        target_length_bias: float = -1.0,
        use_hierarchical_init: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.num_layers = num_layers
        self.target_length_bias = target_length_bias

        # 1. Encoder & Decoder
        self.encoder = MLP(input_dim, hidden_dims, latent_dim, normalize=codebook_normalize)
        self.decoder = MLP(latent_dim, hidden_dims[-1::-1], input_dim, normalize=False)

        # 2. 多层量化器
        self.layers = nn.ModuleList([
            ED_Quantize(
                embed_dim=latent_dim,
                n_embed=codebook_size,
                commitment_weight=commitment_weight,
                forward_mode=codebook_mode,
                codebook_normalize=(i == 0 and codebook_normalize),
                sim_vq=False,
                do_kmeans_init=True,
                entropy_weight=entropy_weight,
                ema_decay=ema_decay,
                use_hierarchical_init=(i == 0 and use_hierarchical_init)
            ) for i in range(num_layers)
        ])

        # 3. 为每一层 (除最后一层外) 实例化一个独立的门控网络
        self.halting_gates  = nn.ModuleList([
            HaltingGate(target_length_bias=target_length_bias) for _ in range(num_layers - 1)
        ])

        # 4. Neighborhood consistency loss module
        self.neighbor_loss_fn = NeighborhoodConsistencyLoss(num_pairs=2048)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x, gumbel_t: float = 0.2):
        B = x.shape[0]
        res = self.encode(x)

        # Save encoder output for neighbor loss (has gradient!)
        z_for_neighbor = res.clone()
        # Save original input for neighbor loss distance computation (no gradient needed)
        x_for_neighbor = x.detach()

        quantize_loss = 0.0
        avg_entropy_val = 0.0

        embs, sem_ids = [], []
        embs_sum = torch.zeros_like(res)

        # 维护一个 Active Mask: 1.0 表示当前层还在量化，0.0 表示已被截断
        active_mask = torch.ones(B, 1, device=x.device)
        # 记录每个样本的真实变长长度
        generated_lengths = torch.zeros(B, device=x.device)

        for i, layer in enumerate(self.layers):
            # 1. 执行当前层的量化
            emb, vq_loss, ent_val, ids = layer(res, temperature=gumbel_t)

            if isinstance(ent_val, torch.Tensor):
                avg_entropy_val += ent_val / self.num_layers
            
            
            # 2. 施加变长 Mask
            masked_emb = emb * active_mask

            embs_sum += masked_emb
            res = res - masked_emb  # 只有 Active 的样本才会扣除残差

            # 3. 累计有效损失 (这里的 vq_loss 现在是纯正数了！)
            vq_loss_masked = vq_loss * active_mask.squeeze()
            quantize_loss += vq_loss_masked.mean()
            generated_lengths += active_mask.squeeze()

            sem_ids.append(ids)
            embs.append(emb)

            # 4. 计算门控截断
            if i < self.num_layers - 1:
                current_energy = (res.detach() ** 2).sum(dim=-1, keepdim=True)
                density = layer.cluster_usage[ids].unsqueeze(1)
                
                # 这里的 logits 已经被 batch_balancer 强制正态化并加上了 bias
                stop_logits = self.halting_gates[i](current_energy, density)

                if self.training:
                    # 使用极其稳定的 RelaxedBernoulli 替代旧版手写的 Gumbel
                    dist = RelaxedBernoulli(temperature=0.5, logits=stop_logits)
                    soft_stop = dist.rsample() 
                    
                    stop_decision = (soft_stop > 0.5).float()
                    stop_decision = stop_decision.detach() - soft_stop.detach() + soft_stop
                
                else:
                    stop_prob = torch.sigmoid(stop_logits)
                    stop_decision = (stop_prob > 0.5).float()
                
                active_mask = active_mask * (1.0 - stop_decision)
            
        # 最终重构
        x_hat = self.decode(embs_sum)

        # 将生成的 List 转为 Tensor
        sem_ids = rearrange(sem_ids, "b d -> d b")

        # 返回丰富的监控与指导信息
        return x_hat, quantize_loss, avg_entropy_val, sem_ids, generated_lengths, z_for_neighbor, x_for_neighbor
    
    def compute_neighbor_loss(self, original_input: torch.Tensor, encoder_output: torch.Tensor,
                              sem_ids: torch.Tensor, gen_lens: torch.Tensor):
        """
        Compute neighborhood consistency loss using original input space distance
        and encoder output gradient.
        
        Args:
            original_input: [B, D_input] raw input embeddings (for distance, no grad)
            encoder_output: [B, D_latent] encoder output (for gradient backprop)
            sem_ids: [B, num_layers] quantized semantic IDs
            gen_lens: [B] actual generated lengths for each sample
        Returns:
            neighbor_loss: scalar loss value
        """
        return self.neighbor_loss_fn(original_input, encoder_output, sem_ids, gen_lens)

    @torch.no_grad()
    def trigger_heuristic_splitting(self, dead_threshold=1.0, gamma=0.01):
        #供训练脚本在外部调用的统一分裂接口
        total_split = 0
        for i, layer in enumerate(self.layers):
            n_split = layer.heuristic_split(dead_threshold=dead_threshold, gamma=gamma)
            total_split += n_split
            if n_split > 0:
                print(f"[Heuristic Split] Layer {i+1} 成功劈开了 {n_split} 个高方差拥挤节点！")
        
        return total_split




