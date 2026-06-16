import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, NamedTuple
from enum import Enum
from einops import rearrange


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
        # 【修改这里】：强制最多只跑 50 次迭代，防止死循环
        out = _Kmeans(k=k, max_iters=50).run(x) 
        tensor.data.copy_(out.centroids)



def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return l2norm(x, dim=self.dim, eps=self.eps)

# ============================================================
# 【新增】对齐源码 distributions/gumbel.py — Gumbel Softmax
# 源码：distributions/gumbel.py
# ============================================================
def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample

# ============================================================
# 【新增】对齐源码 quantize.py — Rotation Trick
# 源码：modules/quantize.py efficient_rotation_trick_transform
# 对应论文 Section 4.2
# ============================================================
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

# ============================================================
# 【新增】对齐源码 quantize.py — 枚举类
# 源码：modules/quantize.py QuantizeForwardMode, QuantizeDistance
# ============================================================
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


# ============================================================
# 【修改】对齐源码 loss.py — QuantizeLoss 独立为类
# 源码：modules/loss.py QuantizeLoss
# ============================================================
class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight=1.0):
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value):
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss



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
        # 【新增】源码 encoder.py L32：末尾追加归一化层
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x):
        assert x.shape[-1] == self.input_dim, \
            f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)

class Quantize(nn.Module):
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
        restart_threshold: float = 0.5,
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
        self.restart_threshold = restart_threshold
        # 注册一个 Buffer，用来记录每个码本节点被激活的指数移动平均次数 (EMA)
        self.register_buffer('cluster_usage', torch.ones(n_embed) * 10.0) # 初始给个安全值 10.0

        # 【新增】源码 quantize.py L70-73
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )

        # 【修改】源码 quantize.py L75：使用独立的 QuantizeLoss 类
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    def _init_weights(self):
        # 源码 quantize.py L86-89：遍历所有模块初始化
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @torch.no_grad()
    def _kmeans_init(self, x):
        print(f"\n[Quantizer] Running KMeans init on GPU (n_embed={self.n_embed})...")
        _kmeans_init_(self.embedding.weight, x.float())
        self.kmeans_initted = True

    # 【新增】源码 quantize.py L96-97
    def get_item_embeddings(self, item_ids):
        return self.out_proj(self.embedding(item_ids))

    # 【修改】forward 新增 temperature 参数，对齐源码 quantize.py L99-156
    def forward(self, x, temperature=0.2):
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        # 【修改】源码 L105：使用 out_proj 处理后的 codebook
        codebook = self.out_proj(self.embedding.weight)

        # 距离计算 — 对齐源码 quantize.py L107-116
        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                (x ** 2).sum(axis=1, keepdim=True) +
                (codebook.T ** 2).sum(axis=0, keepdim=True) -
                2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x / x.norm(dim=1, keepdim=True) @
                (codebook.T) / codebook.T.norm(dim=0, keepdim=True)
            )
        else:
            raise Exception(f"Unsupported Quantize distance mode: {self.distance_mode}")
        _, ids = (dist.detach()).min(axis=1)

        # 计算高熵损失 (Entropy Loss)
        entropy_val = torch.tensor(0.0, device=x.device)
        entropy_penalty = torch.tensor(0.0, device=x.device)
        
        if self.training and self.entropy_weight > 0:
            prob = F.softmax(-dist / temperature, dim=-1)
            avg_prob = prob.mean(dim=0)
            
            # 真实的熵公式：H = -sum(p * log(p))，必为正数！
            entropy_val = -torch.sum(avg_prob * torch.log(avg_prob + 1e-8))
            
            # 为了让优化器最大化正数的熵，我们把它的负数作为惩罚项
            entropy_penalty = -entropy_val * self.entropy_weight

        # 🌟 4. 死码重启机制 (保持你上一版的修复，加上类型转换)
        if self.training and self.restart_threshold > 0:
            with torch.no_grad():
                counts = torch.bincount(ids, minlength=self.n_embed).float()
                self.cluster_usage.mul_(0.99).add_(counts, alpha=0.01)
                dead_mask = self.cluster_usage < self.restart_threshold
                num_dead = dead_mask.sum().item()
                if num_dead > 0:
                    rand_idx = torch.randint(0, x.shape[0], (num_dead,), device=x.device)
                    new_features = x[rand_idx].detach() + torch.randn_like(x[rand_idx]) * 1e-5
                    # 修复 BFloat16 冲突
                    self.embedding.weight.data[dead_mask] = new_features.to(self.embedding.weight.dtype)
                    self.cluster_usage[dead_mask] = 1.0



        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                # 源码 quantize.py L124-129
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=x.device
                )
                emb = weights @ codebook
                emb_out = emb

            elif self.forward_mode == QuantizeForwardMode.STE:
                # 源码 quantize.py L130-132
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()

            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                # 源码 quantize.py L133-142 — 论文核心贡献
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x
                )
                emb_out = emb_out * (
                    torch.norm(emb, dim=1, keepdim=True)
                    / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                ).detach()

            else:
                raise Exception(f"Unsupported Quantize forward mode: {self.forward_mode}")

            # 源码 quantize.py L146：loss 使用 emb (不是 emb_out)
            vq_loss = self.quantize_loss(query=x, value=emb)
            total_vq_loss = vq_loss + entropy_penalty
            return emb_out, total_vq_loss, entropy_val, ids

        else:
            # 源码 quantize.py L148-150：推理时不做 STE
            emb_out = self.get_item_embeddings(ids)
            vq_loss = self.quantize_loss(query=x, value=emb_out)
            return emb_out, vq_loss, entropy_val, ids


# ============================================================
# 【修改】RQVAE — 对齐源码 modules/rqvae.py RqVae
# 关键差异：
#   1. Encoder normalize 跟随 codebook_normalize 参数
#   2. Decoder 始终 normalize=True（源码 L87）
#   3. Quantize 层第 0 层传 codebook_normalize（源码 L70）
#   4. forward 新增 gumbel_t 参数
# ============================================================
class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
        codebook_size: int = 1024,
        num_layers: int = 3,
        commitment_weight: float = 0.25,
        codebook_normalize: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.ROTATION_TRICK,
        entropy_weight: float = 0.1, 
        restart_threshold: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # 源码 rqvae.py L76-81：encoder 的 normalize 跟随 codebook_normalize
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=latent_dim,
            normalize=codebook_normalize
        )

        # 源码 rqvae.py L64-74：每层 Quantize 传 codebook_normalize=(i==0 and codebook_normalize)
        self.layers = nn.ModuleList([
            Quantize(
                embed_dim=latent_dim,
                n_embed=codebook_size,
                commitment_weight=commitment_weight,
                forward_mode=codebook_mode,
                codebook_normalize=(i == 0 and codebook_normalize),
                sim_vq=False,
                do_kmeans_init=True,
                entropy_weight=entropy_weight,
                restart_threshold=restart_threshold,
            ) for i in range(num_layers)
        ])

        # 【关键修改】源码 rqvae.py L83-88：Decoder 始终 normalize=True
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False  # 源码硬编码 True
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    # 【修改】新增 gumbel_t 参数，对齐源码 get_semantic_ids 签名
    def forward(self, x, gumbel_t: float = 0.2):
        res = self.encode(x)
        quantize_loss = 0
        avg_entropy_val = 0.0
        embs, sem_ids = [], []

        for layer in self.layers:
            emb, vq_loss, ent_val, ids = layer(res, temperature=gumbel_t)
            quantize_loss += vq_loss

            if isinstance(ent_val, torch.Tensor):
                avg_entropy_val += ent_val / len(self.layers)

            res = res - emb
            sem_ids.append(ids)
            embs.append(emb)

        embs_sum = torch.stack(embs, dim=0).sum(dim=0)
        x_hat = self.decode(embs_sum)

        # sem_ids: (n_layers, batch) — 与源码 rearrange(sem_ids, "b d -> d b") 一致
        return x_hat, quantize_loss, avg_entropy_val, rearrange(sem_ids, "b d -> d b")