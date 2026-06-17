import sys
import os

# 获取当前 model.py 所在目录的父目录的绝对路径，并加入到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ops.triton.jagged import padded_to_jagged_tensor, jagged_to_flattened_tensor

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根 (Root Mean Square)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class JaggedAttention(nn.Module):
    # 纯粹的 Self-Attention，专供 Encoder 处理 NestedTensor
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # x 此时是 Triton 的 NestedTensor
        queries, keys, values = self.qkv_proj(x).chunk(3, dim=-1)

        queries = queries.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = keys.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = values.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        # FlashAttention 会自动识别 NestedTensor，不需要传 Mask
        attn_out = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).flatten(-2)
        return self.out_proj(attn_out)

class JaggedEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = JaggedAttention(d_model, num_heads)
        self.norm1 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),           # 激活后的 Dropout
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.Dropout(dropout)            # FFN 输出的 Dropout
        )
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x):
        attn_out = self.self_attn(self.norm1(x))
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

class DenseAttention(nn.Module):
    # 支持 Causal Mask 和 Cross Attention，专供 Decoder 处理普通 Tensor
    def __init__(self, d_model, num_heads, is_cross_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_cross_attn = is_cross_attn

        if is_cross_attn:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, context=None, is_causal=False, context_mask=None):
        B, N_q, _ = x.shape

        if self.is_cross_attn:
            queries = self.q_proj(x)
            keys, values = self.kv_proj(context).chunk(2, dim=-1)
        else:
            queries, keys, values = self.qkv_proj(x).chunk(3, dim=-1)

        # 传统的 Reshape 与转置: [Batch, Seq, Heads, Head_dim] -> [Batch, Heads, Seq, Head_dim]
        queries = queries.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            # 扩展为四维以支持多头广播: [Batch, 1, 1, Seq_k]
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2)

        attn_out = F.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.out_proj(attn_out)

class DenseDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = DenseAttention(d_model, num_heads, is_cross_attn=False)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = DenseAttention(d_model, num_heads, is_cross_attn=True)
        self.norm2 = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False), 
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.Dropout(dropout)
        )
        self.norm3 = RMSNorm(d_model)


    def forward(self, x, context, context_mask=None):
        # 1. Masked Self-Attention
        attn_out = self.self_attn(self.norm1(x), is_causal=True)
        x = x + attn_out

        # 2. Cross-Attention
        cross_out = self.cross_attn(self.norm2(x), context=context, context_mask=context_mask)
        x = x + cross_out

        # 3. FFN
        ffn_out = self.ffn(self.norm3(x))
        x = x + ffn_out

        return x

class Tiger(nn.Module):
    def __init__(self, total_vocab_size, vocab_sizes, max_seq_len=64, user_vocab_size=20000, sid_length=4, d_model=64, nhead=8, num_layers=8, dropout=0.1, use_tied_lm_head=True):
        super().__init__()

        self.use_tied_lm_head = use_tied_lm_head
        
        self.sid_length = sid_length
        self.d_model = d_model
        self.total_vocab_size = total_vocab_size

        self.vocab_sizes = vocab_sizes  # e.g. [1024, 768, 512, 1024]
        self.offsets = [0]
        for i in range(len(vocab_sizes) - 1):
            self.offsets.append(self.offsets[-1] + vocab_sizes[i])
        # → offsets = [0, 1024, 1792, 2304]
        
        self.sem_pad_idx = total_vocab_size
        self.sem_emb = nn.Embedding(total_vocab_size + 1, d_model, padding_idx=self.sem_pad_idx)
        self.user_emb = nn.Embedding(user_vocab_size, d_model)
        self.bos_emb = nn.Parameter(torch.rand(1, 1, d_model))
        # 绝对位置编码 (支持到 2048 长)
        max_positions = max_seq_len * sid_length
        #max_positions = max_token_len + 1
        self.enc_wpe = nn.Embedding(max_positions, d_model) # 专供历史记录使用
        self.dec_wpe = nn.Embedding(sid_length, d_model)    # 专供未来预测使用
        self.tte = nn.Embedding(sid_length, d_model)


        self.norm = RMSNorm(d_model)
        self.norm_cxt = RMSNorm(d_model)
        self.input_do = nn.Dropout(p=0.2)
        self.in_proj_context = nn.Linear(d_model, d_model, bias=False)
        self.in_proj = nn.Linear(d_model, d_model, bias=False)

        self.encoder_blocks = nn.ModuleList([
            JaggedEncoderBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers // 2)
        ])
        self.decoder_blocks = nn.ModuleList([
            DenseDecoderBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers // 2)
        ])
        #self.loss_weights = [1.0, 0.8, 0.5, 0.2, 0.05, 0.01]
        # Dynamic loss weights: full weight for all layers, compatible with both fixed-length and variable-length
        self.loss_weights = [1.0] * sid_length
        """
        self.out_proj_layers = nn.ModuleList([
            nn.Linear(d_model, vs, bias=False) for vs in vocab_sizes
        ])
        """
        if not self.use_tied_lm_head:
            self.out_proj_layers = nn.ModuleList([
                nn.Linear(d_model, vs + 1, bias=False) for vs in vocab_sizes
            ])
    
    def forward(self, batch):
        B = batch.user_ids.size(0)
        device = batch.sem_ids.device

        # --- 1. Encoder 阶段 (Token-level，已移除 Item Pooling) ---
        N_enc = batch.sem_ids.size(1)
        enc_sem_ids = batch.sem_ids.clone()
        enc_sem_ids[~batch.seq_mask] = self.sem_pad_idx
        seq_emb = self.sem_emb(enc_sem_ids)
        
        is_new_item = ((batch.token_type_ids == 0) & batch.seq_mask).long()
        item_positions = torch.cumsum(is_new_item, dim=1) - 1 # 形如 [0,0, 1,1,1, 2]

        seq_wpe_emb = self.enc_wpe(item_positions)
        seq_layer_emb = self.tte(batch.token_type_ids)
        seq_emb = seq_emb + seq_wpe_emb + seq_layer_emb
        
        # 直接拼接用户向量与 Token 级序列向量
        u_emb = self.user_emb(batch.user_ids).unsqueeze(1)
        enc_input = torch.cat([u_emb, seq_emb], dim=1) # [B, 1 + N_enc, D]
        
        enc_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            batch.seq_mask
        ], dim=1)

        enc_input = self.in_proj_context(self.input_do(self.norm(enc_input)))

        enc_lengths = enc_mask.sum(dim=1).to(torch.int32)
        enc_jagged = padded_to_jagged_tensor(enc_input, lengths=enc_lengths, max_len=enc_input.size(1))

        for block in self.encoder_blocks:
            enc_jagged = block(enc_jagged)

        # --- 2. 桥接阶段 (解压) ---
        flat_enc = jagged_to_flattened_tensor(enc_jagged)
        enc_dense = torch.zeros(
            B, enc_input.size(1), self.d_model, 
            device=flat_enc.device, dtype=flat_enc.dtype
        )
        enc_dense[enc_mask] = flat_enc

        # --- 3. Decoder 阶段 ---
        fut_emb = self.sem_emb(batch.sem_ids_fut)
        tte_fut = self.tte(torch.arange(self.sid_length, device=fut_emb.device).unsqueeze(0))
        fut_emb = fut_emb + tte_fut
        bos = self.bos_emb.expand(B, -1, -1)
        dec_input = torch.cat([bos, fut_emb[:, :-1, :]], dim=1) # [B, 4, D]
        dec_input = dec_input + self.dec_wpe(torch.arange(self.sid_length, device=dec_input.device).unsqueeze(0))
        dec_input = self.in_proj(self.input_do(self.norm_cxt(dec_input)))
        
        for block in self.decoder_blocks:
            dec_input = block(dec_input, context=enc_dense, context_mask=enc_mask)

        # --- 4. Per-layer prediction & Loss ---
        total_loss = 0.0
        all_logits = []

        total_valid_count = torch.zeros(B, device=device)
        pad_weight = self.sem_emb.weight[self.sem_pad_idx : self.sem_pad_idx + 1, :]

        for layer_idx in range(self.sid_length):
            start_idx = self.offsets[layer_idx]
            local_vs = self.vocab_sizes[layer_idx]
            end_idx = start_idx + local_vs
            
            # ⭐ 按开关分支计算 logits, 输出维度统一为 [B, local_vs + 1]
            if self.use_tied_lm_head:
                # 原逻辑: weight tying + per-layer slicing
                layer_weight = self.sem_emb.weight[start_idx:end_idx, :]
                combined_weight = torch.cat([layer_weight, pad_weight], dim=0)
                layer_logits = F.linear(dec_input[:, layer_idx, :], combined_weight)
            else:
                # 新逻辑: 独立 per-layer Linear (对齐 TIGER 原论文)
                layer_logits = self.out_proj_layers[layer_idx](dec_input[:, layer_idx, :])
            
            all_logits.append(layer_logits)
            
            global_targets = batch.sem_ids_fut[:, layer_idx]
            is_pad = (global_targets == self.sem_pad_idx)
            
            local_vs_tensor = torch.tensor(local_vs, dtype=global_targets.dtype, device=device)
            layer_target = torch.where(is_pad, local_vs_tensor, global_targets - start_idx)
            
            is_current_pad = (global_targets == self.sem_pad_idx)
            is_prev_pad = (batch.sem_ids_fut[:, layer_idx - 1] == self.sem_pad_idx) if layer_idx > 0 else torch.zeros_like(is_current_pad)
            
            layer_valid = (~is_prev_pad).float() 
            
            layer_loss = F.cross_entropy(layer_logits, layer_target, reduction='none')
            layer_loss = layer_loss * layer_valid
            
            weight = self.loss_weights[layer_idx]
            total_loss = total_loss + (layer_loss * weight)
            
            total_valid_count = total_valid_count + layer_valid
        
        total_loss = total_loss / total_valid_count.clamp(min=1.0) 
        loss = total_loss.mean()
        
        return loss, all_logits
    
    @torch.no_grad()
    def generate(self, batch, n_candidates=256, k=64, temperature=1.0):
        # 原始DFA前缀树解码
        self.eval()
        B = batch.user_ids.size(0)
        device = self.dfa_keys.device

        # --- 1. Encoder 阶段 (Token-level) ---
        N_enc = batch.sem_ids.size(1)
        enc_sem_ids = batch.sem_ids.clone()
        enc_sem_ids[~batch.seq_mask] = self.sem_pad_idx
        seq_emb = self.sem_emb(enc_sem_ids)

        is_new_item = ((batch.token_type_ids == 0) & batch.seq_mask).long()
        item_positions = torch.cumsum(is_new_item, dim=1) - 1
        seq_wpe_emb = self.enc_wpe(item_positions)
        seq_layer_emb = self.tte(batch.token_type_ids)
        seq_emb = seq_emb + seq_wpe_emb + seq_layer_emb

        u_emb = self.user_emb(batch.user_ids).unsqueeze(1)
        enc_input = torch.cat([u_emb, seq_emb], dim=1)
        
        enc_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            batch.seq_mask
        ], dim=1)

        enc_input = self.in_proj_context(self.input_do(self.norm(enc_input)))
        enc_lengths = enc_mask.sum(dim=1).to(torch.int32)
        enc_jagged = padded_to_jagged_tensor(enc_input, lengths=enc_lengths, max_len=enc_input.size(1))

        for block in self.encoder_blocks:
            enc_jagged = block(enc_jagged)
        
        flat_enc = jagged_to_flattened_tensor(enc_jagged)
        enc_dense = torch.zeros(
            B, enc_input.size(1), self.d_model, device=device, dtype=flat_enc.dtype
        )
        enc_dense[enc_mask] = flat_enc

        # --- 2. 初始化生成状态 ---
        generated_codes = None
        current_log_probs = torch.zeros(B, 1, device=device) 
        enc_dense_exp = enc_dense
        enc_mask_exp = enc_mask
        
        current_states = torch.zeros(B, 1, dtype=torch.long, device=device)
        
        # 🌟 关键防御：冻结掩码初始化
        frozen_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        # --- 3. 自回归生成 ---
        for step in range(self.sid_length):
            current_k = 1 if step == 0 else k

            if step == 1:
                enc_dense_exp = enc_dense.repeat_interleave(k, dim=0)
                enc_mask_exp = enc_mask.repeat_interleave(k, dim=0)
            
            # 全局早停
            if step > 0 and frozen_mask.all():
                remaining_pad = torch.full(
                    (B, k, self.sid_length - step), self.sem_pad_idx,
                    dtype=generated_codes.dtype, device=device
                )
                generated_codes = torch.cat([generated_codes, remaining_pad], dim=-1)
                break
            
            bos = self.bos_emb.expand(B * current_k, -1, -1)

            if step == 0:
                dec_input = bos
            else:
                flat_gen = generated_codes.view(B * current_k, step)
                fut_emb = self.sem_emb(flat_gen)
                tte_fut = self.tte(torch.arange(step, device=device).unsqueeze(0))
                fut_emb = fut_emb + tte_fut
                dec_input = torch.cat([bos, fut_emb], dim=1)
            
            dec_input = dec_input + self.dec_wpe(torch.arange(dec_input.size(1), device=device).unsqueeze(0))
            dec_input = self.in_proj(self.input_do(self.norm_cxt(dec_input)))

            for block in self.decoder_blocks:
                dec_input = block(dec_input, context=enc_dense_exp, context_mask=enc_mask_exp)
            
            dec_last = dec_input[:, -1, :]  
            
            start_idx = self.offsets[step]
            local_vs = self.vocab_sizes[step]
            end_idx = start_idx + local_vs
            
            # ⭐ 按开关分支计算 logits, 输出维度统一为 [B*k, local_vs + 1]
            if self.use_tied_lm_head:
                # 原逻辑: weight tying + per-layer slicing
                layer_weight = self.sem_emb.weight[start_idx:end_idx, :]
                pad_weight = self.sem_emb.weight[self.sem_pad_idx : self.sem_pad_idx + 1, :]
                combined_weight = torch.cat([layer_weight, pad_weight], dim=0)
                logits = F.linear(dec_last, combined_weight)
            else:
                # 新逻辑: 独立 per-layer Linear
                logits = self.out_proj_layers[step](dec_last)  

            # --- DFA 安全校验 ---
            tokens_to_check = torch.cat([
                torch.arange(start_idx, end_idx, device=device),
                torch.tensor([self.sem_pad_idx], device=device)
            ])
            
            query_keys = current_states.view(-1).unsqueeze(1) * self.dfa_multiplier + tokens_to_check.unsqueeze(0)
            idx = torch.searchsorted(self.dfa_keys, query_keys).clamp(max=self.dfa_keys.size(0) - 1)
            valid_mask = self.dfa_keys[idx] == query_keys
            
            # 🌟 豁免冻结路径的 DFA 校验
            if step > 0:
                frozen_flat = frozen_mask.view(-1)
                valid_mask[frozen_flat, :] = True
            
            logits = logits.masked_fill(~valid_mask, -10000.0)
            
            # 🌟 强制冻结路径 100% 预测 PAD
            if step > 0:
                frozen_flat = frozen_mask.view(-1)
                if frozen_flat.any():
                    force_pad_mask = torch.zeros_like(logits, dtype=torch.bool)
                    force_pad_mask[frozen_flat, :-1] = True
                    logits = logits.masked_fill(force_pad_mask, -10000.0)
                    logits[frozen_flat, -1] = 0.0

            logits = logits.view(B, current_k, -1)
            probs = F.softmax(logits / temperature, dim=-1)
            probs_flat = probs.view(B * current_k, -1)
            
            # 采样
            sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=n_candidates, replacement=False)
            sampled_probs_flat = torch.gather(probs_flat, 1, sampled_tokens_flat)
            sampled_log_probs_flat = torch.log(sampled_probs_flat + 1e-9)
            
            # 局部转全局 ID
            is_sampled_pad = (sampled_tokens_flat == local_vs)
            pad_tensor = torch.tensor(self.sem_pad_idx, dtype=sampled_tokens_flat.dtype, device=device)
            sampled_tokens_global_flat = torch.where(
                is_sampled_pad, pad_tensor, sampled_tokens_flat + start_idx         
            )
            
            sampled_tokens = sampled_tokens_global_flat.view(B, current_k, n_candidates)
            sampled_log_probs = sampled_log_probs_flat.view(B, current_k, n_candidates)

            # 拼接路径
            if step == 0:
                all_paths = sampled_tokens.view(B, n_candidates, 1)
                path_log_probs = sampled_log_probs.view(B, n_candidates) 
            else:
                expanded_history = generated_codes.unsqueeze(2).expand(-1, -1, n_candidates, -1)
                new_tokens = sampled_tokens.unsqueeze(-1) 
                all_paths = torch.cat([expanded_history, new_tokens], dim=-1).view(B, current_k * n_candidates, step + 1)
                
                expanded_log_probs = current_log_probs.unsqueeze(-1) + sampled_log_probs
                path_log_probs = expanded_log_probs.view(B, current_k * n_candidates)
            
            # 🌟 绝对纯净的 Top-K 选拔
            next_k = min(k, path_log_probs.shape[1])
            top_log_probs, top_indices = torch.topk(path_log_probs, next_k, dim=-1)
            current_log_probs = top_log_probs

            # 更新生成的路径
            generated_codes = torch.gather(
                all_paths, 
                1, 
                top_indices.unsqueeze(-1).expand(-1, -1, step + 1)
            )
                
            # --- 状态机转移 ---
            kept_tokens_global = generated_codes[:, :, -1].view(-1)
            parent_indices = top_indices // n_candidates
            batch_offsets = torch.arange(B, device=device).unsqueeze(1) * current_k
            parent_states_flat = current_states.view(-1)[(batch_offsets + parent_indices).view(-1)]
            
            transition_queries = parent_states_flat * self.dfa_multiplier + kept_tokens_global
            idx_next = torch.searchsorted(self.dfa_keys, transition_queries).clamp(max=self.dfa_keys.size(0) - 1)
            current_states = self.dfa_values[idx_next].view(B, next_k)
            
            # 如果当前是 PAD，状态归 0 锁死
            is_pad_now = (kept_tokens_global == self.sem_pad_idx).view(B, next_k)
            current_states[is_pad_now] = 0
            
            # PAD 物理传播
            if step > 0:
                prev_tokens = generated_codes[:, :, step - 1]  
                already_stopped = (prev_tokens == self.sem_pad_idx)  
                generated_codes[:, :, step][already_stopped] = self.sem_pad_idx
            
            # 🌟 极其致命的遗漏修复：必须在每步最后更新 frozen_mask！
            current_last_tokens = generated_codes[:, :, step]
            frozen_mask = (current_last_tokens == self.sem_pad_idx)
        
        return generated_codes, current_log_probs
    

    # 稀疏状态机离线构建
    @torch.no_grad()
    def build_trie_dfa(self, valid_codes_tensor):
        """
        构建显存极度友好的 Sparse DFA (仅消耗 ~20MB 显存)
        完全避免了稠密矩阵的 80GB OOM 问题。
        """
        print("\n[*] 正在离线构建 稀疏 DFA 状态机...")
        device = valid_codes_tensor.device
        
        valid_codes_list = valid_codes_tensor.cpu().tolist()
        
        trie = {0: {}}
        state_counter = 1

        for code in valid_codes_list:
            current_state = 0
            for token in code:
                if token not in trie[current_state]:
                    trie[current_state][token] = state_counter
                    trie[state_counter] = {}
                    state_counter += 1
                current_state = trie[current_state][token]

        print(f"[*] DFA 构建完成: 共 {state_counter} 个独立状态")
        
        keys = []
        next_states = []
        
        # 使用一个足够大的倍数将 (state, token) 压成一个 64位 整数
        # state 最大约 2,000,000, token 最大约 6000。100,000 倍数绝对安全，不会溢出
        MULTIPLIER = 100000 
        
        for state, transitions in trie.items():
            for token, next_state in transitions.items():
                keys.append(state * MULTIPLIER + token)
                next_states.append(next_state)
                
        keys_tensor = torch.tensor(keys, dtype=torch.long, device=device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.long, device=device)
        
        # 按照 Key 排序，这是二分查找 (searchsorted) 的核心前提
        sorted_indices = torch.argsort(keys_tensor)
        self.dfa_keys = keys_tensor[sorted_indices]
        self.dfa_values = next_states_tensor[sorted_indices]
        self.dfa_multiplier = MULTIPLIER
        print(f"[*] 稀疏矩阵转换完毕: 边总数 {len(keys)}。")

    """
    @torch.no_grad()
    def _check_valid_prefix(self, prefixes, valid_prefixes, chunk_size=512):
        total_paths = prefixes.shape[0]
        matches = torch.zeros(total_paths, dtype=torch.bool, device=prefixes.device)

        for i in range(0, total_paths, chunk_size):
            chunk = prefixes[i : i + chunk_size] # [chunk, seq_len]
            # 广播比对逻辑：[chunk, 1, seq_len] == [1, N_items, seq_len] -> [chunk, N_items, seq_len]
            match_chunk = (chunk.unsqueeze(1) == valid_prefixes.unsqueeze(0)).all(dim=-1).any(dim=-1)
            matches[i : i + chunk_size] = match_chunk
        
        return matches
    """
    @torch.no_grad()
    # 将默认的 chunk_size 从 512 直接开到 8192 甚至更大！
    def _check_valid_prefix(self, prefixes, valid_prefixes, chunk_size=8192):
        total_paths = prefixes.shape[0]
        matches = torch.zeros(total_paths, dtype=torch.bool, device=prefixes.device)

        for i in range(0, total_paths, chunk_size):
            chunk = prefixes[i : i + chunk_size] # [chunk, seq_len]
            # 广播比对逻辑
            match_chunk = (chunk.unsqueeze(1) == valid_prefixes.unsqueeze(0)).all(dim=-1).any(dim=-1)
            matches[i : i + chunk_size] = match_chunk
        
        return matches
    
    @torch.no_grad()
    def generate_full(self, batch, valid_codes_tensor, n_candidates=256, k=64, temperature=1.0):
        """
        暴力检索比对字典，速度慢
        受限解码生成入口，完美适配 Tiger 的 4 位 SID (sid_length=4)
        Multinomial采样,极致缓存优化
        :param batch: 原始输入 Batch (包含 user_ids, sem_ids, token_type_ids 等)
        :param valid_codes_tensor: 全局字典矩阵 [Num_Items, 4], 设备需要和模型一致
        :param n_candidates: 每次预测从 Logits 里初筛多少个选项
        :param k: 经历校验惩罚后，最终保留多少条活着的分支
        :return: (最佳合法路径矩阵 [B, k, sid_length], 对应的概率 [B, k])
        """
        self.eval()
        B = batch.user_ids.size(0)
        device = valid_codes_tensor.device

        # 1. 独立运行一次 Encoder 获取全量上下文
        N_enc = batch.sem_ids.size(1)
        enc_sem_ids = batch.sem_ids.clone()
        enc_sem_ids[~batch.seq_mask] = self.sem_pad_idx
        seq_emb = self.sem_emb(enc_sem_ids)

        # 【同步修复】
        is_new_item = ((batch.token_type_ids == 0) & batch.seq_mask).long()
        item_positions = torch.cumsum(is_new_item, dim=1) - 1
        
        seq_wpe_emb = self.enc_wpe(item_positions)
        seq_layer_emb = self.tte(batch.token_type_ids)
        seq_emb = seq_emb + seq_wpe_emb + seq_layer_emb

        u_emb = self.user_emb(batch.user_ids).unsqueeze(1)
        enc_input = torch.cat([u_emb, seq_emb], dim=1)
        enc_input = self.in_proj_context(self.input_do(self.norm(enc_input)))

        enc_lengths = batch.seq_mask.sum(dim=1).to(torch.int32) + 1 
        enc_jagged = padded_to_jagged_tensor(enc_input, lengths=enc_lengths, max_len=enc_input.size(1))

        for block in self.encoder_blocks:
            enc_jagged = block(enc_jagged)
        
        enc_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            batch.seq_mask
        ], dim=1)
        flat_enc = jagged_to_flattened_tensor(enc_jagged)
        enc_dense = torch.zeros(
            B, enc_input.size(1), self.d_model, device=device, dtype=flat_enc.dtype
        )
        enc_dense[enc_mask] = flat_enc

        # 2. 初始化 Beam Search 状态变量
        generated_codes = None
        current_log_probs = torch.zeros(B, 1, device=device) # 初始根节点概率为 0

        enc_dense_exp = enc_dense
        enc_mask_exp = enc_mask

        # 3. 开启自回归循环生成 4 位 Semantic ID
        for step in range(self.sid_length):
            # 第一步强制 batch_size 就是原始的 B，后续裂变为 B * K
            current_k = 1 if step == 0 else k

            if step == 1:
                enc_dense_exp = enc_dense.repeat_interleave(k, dim=0)
                enc_mask_exp = enc_mask.repeat_interleave(k, dim=0)
            
            bos = self.bos_emb.expand(B * current_k, -1, -1)

            if step == 0:
                dec_input = bos
            else:
                # 把已经生成的路径当做输入喂给 Decoder
                flat_gen = generated_codes.view(B * current_k, step)
                # generated_codes 中的值已在全局 offset 空间，直接查 sem_emb
                fut_emb = self.sem_emb(flat_gen)
                # tte only added to sem_emb, NOT to bos (matching source code)
                tte_fut = self.tte(torch.arange(step, device=device).unsqueeze(0))
                fut_emb = fut_emb + tte_fut
                dec_input = torch.cat([bos, fut_emb], dim=1)
            
            dec_input = dec_input + self.dec_wpe(torch.arange(dec_input.size(1), device=device).unsqueeze(0))
            dec_input = self.in_proj(self.input_do(self.norm_cxt(dec_input)))

            # 跑一轮 Decoder 的全层网络
            for block in self.decoder_blocks:
                dec_input = block(dec_input, context=enc_dense_exp, context_mask=enc_mask_exp)
            
            # 提取时间步最后一个 token 的预测向量
            dec_last = dec_input[:, -1, :]  # [B * current_k, D]
            
            # 【修改 3】：拼接推理期的权重矩阵
            start_idx = self.offsets[step]
            end_idx = start_idx + self.vocab_sizes[step]
            layer_weight = self.sem_emb.weight[start_idx:end_idx, :]
            pad_weight = self.sem_emb.weight[self.sem_pad_idx : self.sem_pad_idx + 1, :]
            combined_weight = torch.cat([layer_weight, pad_weight], dim=0)
            
            # 计算 Logits (词表大小变为了 vocab_sizes[step] + 1)
            logits = F.linear(dec_last, combined_weight)  
            logits = logits.view(B, current_k, -1)
            probs = F.softmax(logits / temperature, dim=-1)

            # 将多维拉平，满足 multinomial 对 2D 张量的要求
            probs_flat = probs.view(B * current_k, -1)
            
            # 1. 采样
            sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=n_candidates, replacement=False)
            sampled_probs_flat = torch.gather(probs_flat, 1, sampled_tokens_flat)
            sampled_log_probs_flat = torch.log(sampled_probs_flat + 1e-9)
            
            # 2. 局部索引转回全局索引 (极其关键！)
            local_vs = self.vocab_sizes[step]
            is_sampled_pad = (sampled_tokens_flat == local_vs)
            
            pad_tensor = torch.tensor(self.sem_pad_idx, dtype=sampled_tokens_flat.dtype, device=device)
            sampled_tokens_flat = torch.where(
                is_sampled_pad, 
                pad_tensor,                             # 如果模型采样到了最后的 PAD 位，还原为全局 2048
                sampled_tokens_flat + start_idx         # 否则还原为带 offset 的全局正常 ID
            )
            
            sampled_tokens = sampled_tokens_flat.view(B, current_k, n_candidates)
            sampled_log_probs = sampled_log_probs_flat.view(B, current_k, n_candidates)


            # 构造要送去“暴力校验”的完整前缀组合
            if step == 0:
                prefixes_to_check = sampled_tokens.view(B, n_candidates, 1) 
                path_log_probs = sampled_log_probs.view(B, n_candidates) 
            else:
                expanded_history = generated_codes.unsqueeze(2).repeat(1, 1, n_candidates, 1) 
                new_tokens = sampled_tokens.unsqueeze(-1) 
                prefixes_to_check = torch.cat([expanded_history, new_tokens], dim=-1) 
                
                prefixes_to_check = prefixes_to_check.view(B, current_k * n_candidates, step + 1)
                expanded_log_probs = current_log_probs.unsqueeze(-1) + sampled_log_probs
                path_log_probs = expanded_log_probs.view(B, current_k * n_candidates)
            
            # 抽取合法字典的对应前缀长度，分块暴力比对
            valid_prefixes = valid_codes_tensor[:, :step + 1]
            flat_prefixes = prefixes_to_check.view(-1, step + 1)

            matches = self._check_valid_prefix(flat_prefixes, valid_prefixes, chunk_size=512)
            matches = matches.view(B, -1)

            # 非法路径赋予 -10000.0 的物理超度
            penalized_log_probs = path_log_probs + (-10000.0 * (~matches).float())

            # 再次精筛，只保留活下来的 Top K 条路线
            next_k = min(k, penalized_log_probs.shape[1])
            top_log_probs, top_indices = torch.topk(penalized_log_probs, next_k, dim=-1)

            # 更新状态，进入下一步
            generated_codes = torch.gather(
                prefixes_to_check,
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, step + 1)
            )
            current_log_probs = top_log_probs

            # PAD 传播：如果上一步已经是 PAD，强制当前步也为 PAD
            # 保证变长 SID 一旦终止就不会再生成有效 token
            if step > 0:
                prev_tokens = generated_codes[:, :, step - 1]  # [B, k]
                already_stopped = (prev_tokens == self.sem_pad_idx)  # [B, k]
                generated_codes[:, :, step][already_stopped] = self.sem_pad_idx
        
        return generated_codes, current_log_probs
    
    @torch.no_grad()
    def generate_unique(self, batch, valid_codes_tensor, n_candidates=256, k=64, temperature=1.0):
        """
        极速去重补丁，比dfa慢，但比暴力检索快
        受限解码生成入口，完美适配 Tiger 的 4 位 SID (sid_length=4)
        Multinomial采样,极致缓存优化
        """
        self.eval()
        B = batch.user_ids.size(0)
        device = valid_codes_tensor.device

        # 1. 独立运行一次 Encoder 获取全量上下文
        N_enc = batch.sem_ids.size(1)
        enc_sem_ids = batch.sem_ids.clone()
        enc_sem_ids[~batch.seq_mask] = self.sem_pad_idx
        seq_emb = self.sem_emb(enc_sem_ids)

        # 【同步修复】
        is_new_item = ((batch.token_type_ids == 0) & batch.seq_mask).long()
        item_positions = torch.cumsum(is_new_item, dim=1) - 1
        
        seq_wpe_emb = self.enc_wpe(item_positions)
        seq_layer_emb = self.tte(batch.token_type_ids)
        seq_emb = seq_emb + seq_wpe_emb + seq_layer_emb

        u_emb = self.user_emb(batch.user_ids).unsqueeze(1)
        enc_input = torch.cat([u_emb, seq_emb], dim=1)
        enc_input = self.in_proj_context(self.input_do(self.norm(enc_input)))

        enc_lengths = batch.seq_mask.sum(dim=1).to(torch.int32) + 1 
        enc_jagged = padded_to_jagged_tensor(enc_input, lengths=enc_lengths, max_len=enc_input.size(1))

        for block in self.encoder_blocks:
            enc_jagged = block(enc_jagged)
        
        enc_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            batch.seq_mask
        ], dim=1)
        flat_enc = jagged_to_flattened_tensor(enc_jagged)
        enc_dense = torch.zeros(
            B, enc_input.size(1), self.d_model, device=device, dtype=flat_enc.dtype
        )
        enc_dense[enc_mask] = flat_enc

        # 2. 初始化 Beam Search 状态变量
        generated_codes = None
        current_log_probs = torch.zeros(B, 1, device=device) # 初始根节点概率为 0

        enc_dense_exp = enc_dense
        enc_mask_exp = enc_mask

        # 3. 开启自回归循环生成 4 位 Semantic ID
        for step in range(self.sid_length):
            # 第一步强制 batch_size 就是原始的 B，后续裂变为 B * K
            current_k = 1 if step == 0 else k

            if step == 1:
                enc_dense_exp = enc_dense.repeat_interleave(k, dim=0)
                enc_mask_exp = enc_mask.repeat_interleave(k, dim=0)
            
            bos = self.bos_emb.expand(B * current_k, -1, -1)

            if step == 0:
                dec_input = bos
            else:
                # 把已经生成的路径当做输入喂给 Decoder
                flat_gen = generated_codes.view(B * current_k, step)
                # generated_codes 中的值已在全局 offset 空间，直接查 sem_emb
                fut_emb = self.sem_emb(flat_gen)
                # tte only added to sem_emb, NOT to bos (matching source code)
                tte_fut = self.tte(torch.arange(step, device=device).unsqueeze(0))
                fut_emb = fut_emb + tte_fut
                dec_input = torch.cat([bos, fut_emb], dim=1)
            
            dec_input = dec_input + self.dec_wpe(torch.arange(dec_input.size(1), device=device).unsqueeze(0))
            dec_input = self.in_proj(self.input_do(self.norm_cxt(dec_input)))

            # 跑一轮 Decoder 的全层网络
            for block in self.decoder_blocks:
                dec_input = block(dec_input, context=enc_dense_exp, context_mask=enc_mask_exp)
            
            # 提取时间步最后一个 token 的预测向量
            dec_last = dec_input[:, -1, :]  # [B * current_k, D]
            
            # 拼接推理期的权重矩阵
            start_idx = self.offsets[step]
            end_idx = start_idx + self.vocab_sizes[step]
            layer_weight = self.sem_emb.weight[start_idx:end_idx, :]
            pad_weight = self.sem_emb.weight[self.sem_pad_idx : self.sem_pad_idx + 1, :]
            combined_weight = torch.cat([layer_weight, pad_weight], dim=0)
            
            # 计算 Logits (词表大小变为了 vocab_sizes[step] + 1)
            logits = F.linear(dec_last, combined_weight)  
            logits = logits.view(B, current_k, -1)
            probs = F.softmax(logits / temperature, dim=-1)

            # 将多维拉平，满足 multinomial 对 2D 张量的要求
            probs_flat = probs.view(B * current_k, -1)
            
            # 1. 采样
            sampled_tokens_flat = torch.multinomial(probs_flat, num_samples=n_candidates, replacement=False)
            sampled_probs_flat = torch.gather(probs_flat, 1, sampled_tokens_flat)
            sampled_log_probs_flat = torch.log(sampled_probs_flat + 1e-9)
            
            # 2. 局部索引转回全局索引 (极其关键！)
            local_vs = self.vocab_sizes[step]
            is_sampled_pad = (sampled_tokens_flat == local_vs)
            
            pad_tensor = torch.tensor(self.sem_pad_idx, dtype=sampled_tokens_flat.dtype, device=device)
            sampled_tokens_flat = torch.where(
                is_sampled_pad, 
                pad_tensor,                             # 如果模型采样到了最后的 PAD 位，还原为全局 2048
                sampled_tokens_flat + start_idx         # 否则还原为带 offset 的全局正常 ID
            )
            
            sampled_tokens = sampled_tokens_flat.view(B, current_k, n_candidates)
            sampled_log_probs = sampled_log_probs_flat.view(B, current_k, n_candidates)


            # 构造要送去“暴力校验”的完整前缀组合
            if step == 0:
                prefixes_to_check = sampled_tokens.view(B, n_candidates, 1) 
                path_log_probs = sampled_log_probs.view(B, n_candidates) 
            else:
                expanded_history = generated_codes.unsqueeze(2).repeat(1, 1, n_candidates, 1) 
                new_tokens = sampled_tokens.unsqueeze(-1) 
                prefixes_to_check = torch.cat([expanded_history, new_tokens], dim=-1) 
                
                prefixes_to_check = prefixes_to_check.view(B, current_k * n_candidates, step + 1)
                expanded_log_probs = current_log_probs.unsqueeze(-1) + sampled_log_probs
                path_log_probs = expanded_log_probs.view(B, current_k * n_candidates)
            
            # 【核心极速去重补丁】
            # 抽取合法字典的对应前缀长度，分块暴力比对
            valid_prefixes = valid_codes_tensor[:, :step + 1]
            
            # 在这里加上 unique！能把几百万行的矩阵瞬间降维到几千甚至几百行！
            unique_valid_prefixes = torch.unique(valid_prefixes, dim=0)
            
            flat_prefixes = prefixes_to_check.view(-1, step + 1)

            # 把 unique_valid_prefixes 传给 _check_valid_prefix
            matches = self._check_valid_prefix(flat_prefixes, unique_valid_prefixes, chunk_size=8192)
            matches = matches.view(B, -1)
            # 【补丁结束】

            # 非法路径赋予 -10000.0 的物理超度
            penalized_log_probs = path_log_probs + (-10000.0 * (~matches).float())

            # 再次精筛，只保留活下来的 Top K 条路线
            next_k = min(k, penalized_log_probs.shape[1])
            top_log_probs, top_indices = torch.topk(penalized_log_probs, next_k, dim=-1)

            # 更新状态，进入下一步
            generated_codes = torch.gather(
                prefixes_to_check,
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, step + 1)
            )
            current_log_probs = top_log_probs

            # PAD 传播：如果上一步已经是 PAD，强制当前步也为 PAD
            # 保证变长 SID 一旦终止就不会再生成有效 token
            if step > 0:
                prev_tokens = generated_codes[:, :, step - 1]  # [B, k]
                already_stopped = (prev_tokens == self.sem_pad_idx)  # [B, k]
                generated_codes[:, :, step][already_stopped] = self.sem_pad_idx
        
        return generated_codes, current_log_probs

