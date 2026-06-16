import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import NamedTuple
from torch import Tensor

class TigerDataset(Dataset):
    def __init__(self, seqs_file, mapping_file, tiger_code_file, meta_file, max_seq_len=20, split='train'):
        print(f"加载数据中 ({split} split)...")
        with open(seqs_file, 'r') as f:
            self.all_seqs = json.load(f)
        with open(mapping_file, 'r') as f:
            self.id_mapping = json.load(f)
        with open(tiger_code_file, 'r') as f:
            self.item2code = json.load(f)
        
        # 🌟 读取 Meta 信息，实现全动态配置！
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        self.user2id = self.id_mapping['user2id']
        self.item2id = self.id_mapping['item2id']
        
        self.max_seq_len = max_seq_len
        self.sid_length = self.meta["sid_length"] # 动态读取 SID 长度 (4)
        
        # 🌟 核心算法：动态计算词表偏移量 (Offsets)
        vocab_sizes = [
            self.meta[f"vocab_size_layer{i+1}"] for i in range(self.sid_length)
        ]
        
        self.offsets = [0]
        for i in range(len(vocab_sizes) - 1):
            self.offsets.append(self.offsets[-1] + vocab_sizes[i])
            
        self.total_vocab_size = sum(vocab_sizes) # 自动算出 3328
        
        print(f"[*] 动态加载配置成功！SID 长度: {self.sid_length}")
        print(f"[*] 各层词表大小: {vocab_sizes}")
        print(f"[*] 自动计算的偏移量: {self.offsets}")
        print(f"[*] 全局总词表大小: {self.total_vocab_size}")
        
        self.samples = []
        for user_str, item_str_list in self.all_seqs.items():
            # 严谨划分：序列至少需要 4 个物品 (1个输入 + 1个Train预测 + 1个Val预测 + 1个Test预测)
            if len(item_str_list) < 4:
                continue
                
            user_int = self.user2id[user_str]
            item_int_str_list = [str(self.item2id[i]) for i in item_str_list]
            
            if split == 'test':
                # 测试集：严格用 1~(N-1) 预测 N
                seq_items = item_int_str_list[:-1]
                target_item = item_int_str_list[-1]
                self.samples.append((user_int, seq_items[-self.max_seq_len:], target_item))
            elif split == 'val':
                # 验证集：严格用 1~(N-2) 预测 N-1
                seq_items = item_int_str_list[:-2]
                target_item = item_int_str_list[-2]
                self.samples.append((user_int, seq_items[-self.max_seq_len:], target_item))
            elif split == 'train':
                # 训练集：使用 N-2 之前的所有数据,滑动窗口数据增强
                train_list = item_int_str_list[:-2]

                # 遍历 train_list，生成多个训练样本
                for i in range(1, len(train_list)):
                    seq_items = train_list[:i]
                    target_item = train_list[i]
                    
                    # 截断超长序列
                    seq_items = seq_items[-self.max_seq_len:]
                    self.samples.append((user_int, seq_items, target_item))
            else:
                raise ValueError("split 必须是 'train', 'val', 或 'test'")

            
        print(f"[*] 成功构建 {len(self.samples)} 条 {split} 样本。")

    def __len__(self):
        return len(self.samples)
        
    def _get_offset_sid(self, item_id_str):
        raw_code = self.item2code[item_id_str]
        return [c + o for c, o in zip(raw_code, self.offsets)]

    def __getitem__(self, idx):
        user_int, seq_items, target_item = self.samples[idx]
        
        flat_seq = []
        for item in seq_items:
            flat_seq.extend(self._get_offset_sid(item))
            
        target_sid = self._get_offset_sid(target_item)
        
        return torch.tensor(user_int, dtype=torch.long), torch.tensor(flat_seq, dtype=torch.long), torch.tensor(target_sid, dtype=torch.long)

    def collate_fn(self, batch):
        # 将样本组装成 TokenizedSeqBatch，并动态处理 Padding 和 Token Type IDs
        user_ids, seqs, target_sids = zip(*batch)
        B = len(batch)

        user_ids = torch.stack(user_ids)
        sem_ids_fut = torch.stack(target_sids)

        PAD_TOKEN_ID = self.total_vocab_size
        
        sem_ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID) 
        seq_mask = (sem_ids != PAD_TOKEN_ID)

        N_tokens = sem_ids.shape[1]
        base_types = torch.arange(self.sid_length)
        repeats = (N_tokens + self.sid_length - 1) // self.sid_length
        token_type_ids = base_types.repeat(repeats)[:N_tokens].unsqueeze(0).repeat(B, 1)
        
        # 将 padding 处的 type_id 强行置零（更严谨）
        token_type_ids = token_type_ids.masked_fill(~seq_mask, 0)
        
        token_type_ids_fut = torch.arange(self.sid_length).unsqueeze(0).repeat(B, 1)
        
        return TokenizedSeqBatch(
            user_ids=user_ids, sem_ids=sem_ids, sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask, token_type_ids=token_type_ids, token_type_ids_fut=token_type_ids_fut
        )


class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor



class VarlenTigerDataset(Dataset):
    def __init__(self, seqs_file, mapping_file, tiger_code_file, meta_file, max_seq_len=50, split='train'):
        print(f"加载变长数据中 ({split} split)...")
        with open(seqs_file, 'r') as f:
            self.all_seqs = json.load(f)
        with open(mapping_file, 'r') as f:
            self.id_mapping = json.load(f)
            
        # 🌟 Varlen 专属修改 1：同时读取 code 和 length 文件
        with open(tiger_code_file, 'r') as f:
            self.item2code = json.load(f)
        #length_file = tiger_code_file.replace("tiger_item2code.json", "tiger_item2length.json")
        #with open(length_file, 'r') as f:
        #    self.item2length = json.load(f)
        
        # 读取 Meta 信息，实现全动态配置！
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        self.user2id = self.id_mapping['user2id']
        self.item2id = self.id_mapping['item2id']
        
        self.max_seq_len = max_seq_len
        self.sid_length = self.meta["sid_length"] # 动态读取最大 SID 长度 (如 6)
        
        # 动态计算词表偏移量 (Offsets)
        vocab_sizes = [
            self.meta[f"vocab_size_layer{i+1}"] for i in range(self.sid_length)
        ]
        
        self.offsets = [0]
        for i in range(len(vocab_sizes) - 1):
            self.offsets.append(self.offsets[-1] + vocab_sizes[i])
            
        self.total_vocab_size = sum(vocab_sizes) 
        
        print(f"[*] 动态加载变长配置成功！最大 SID 长度: {self.sid_length}")
        print(f"[*] 各层词表大小 (包含 LAYER_PAD): {vocab_sizes}")
        print(f"[*] 自动计算的偏移量: {self.offsets}")
        print(f"[*] 全局总词表大小 (含序列 PAD): {self.total_vocab_size}")
        
        self.samples = []
        for user_str, item_str_list in self.all_seqs.items():
            # 严谨划分：序列至少需要 4 个物品
            if len(item_str_list) < 4:
                continue
                
            user_int = self.user2id[user_str]
            item_int_str_list = [str(self.item2id[i]) for i in item_str_list]
            
            if split == 'test':
                seq_items = item_int_str_list[:-1]
                target_item = item_int_str_list[-1]
                self.samples.append((user_int, seq_items[-self.max_seq_len:], target_item))
            elif split == 'val':
                seq_items = item_int_str_list[:-2]
                target_item = item_int_str_list[-2]
                self.samples.append((user_int, seq_items[-self.max_seq_len:], target_item))
            elif split == 'train':
                train_list = item_int_str_list[:-2]
                for i in range(1, len(train_list)):
                    seq_items = train_list[:i]
                    target_item = train_list[i]
                    
                    # 截断超长序列
                    seq_items = seq_items[-self.max_seq_len:]
                    self.samples.append((user_int, seq_items, target_item))
            else:
                raise ValueError("split 必须是 'train', 'val', 或 'test'")
            
        print(f"[*] 成功构建 {len(self.samples)} 条 {split} 样本。")

    def __len__(self):
        return len(self.samples)
        
    def _get_offset_sid_history(self, item_id_str):
        """专供历史序列：仅保留有效长度，抛弃填充码，实现变长极致压缩！"""
        raw_code = self.item2code[item_id_str]
        
        # 【修改这里】：因为是变长字典，字典里的列表长度就是真实长度，直接用 len() 即可！
        actual_len = len(raw_code)
        
        # 仅取前 actual_len 个特征，并加上全局偏移量
        codes = [raw_code[i] + self.offsets[i] for i in range(actual_len)]
        # 因为变长，我们需要同步返回对应的层级 ID (0, 1, 2...)
        types = list(range(actual_len))
        
        return codes, types
    
    def _get_offset_sid_target(self, item_id_str):
        """专供预测目标：强行补齐到最大长度 sid_length，用全局 PAD_TOKEN_ID 填充"""
        raw_code = self.item2code[item_id_str]
        actual_len = len(raw_code)
        
        # 1. 转换实际存在的 Token (加上 offset)
        padded_code = [raw_code[i] + self.offsets[i] for i in range(actual_len)]
        
        # 2. 不足 sid_length 的部分，用全局 PAD_TOKEN_ID 强行补齐
        # 确保 DataLoader 吐出的目标维度严格是 [sid_length]
        pad_token_id = self.total_vocab_size
        while len(padded_code) < self.sid_length:
            padded_code.append(pad_token_id)
            
        return padded_code

    def __getitem__(self, idx):
        # 保持不变
        user_int, seq_items, target_item = self.samples[idx]
        
        flat_seq = []
        flat_types = []
        for item in seq_items:
            codes, types = self._get_offset_sid_history(item)
            flat_seq.extend(codes)
            flat_types.extend(types)
            
        target_sid = self._get_offset_sid_target(target_item)
        
        return (torch.tensor(user_int, dtype=torch.long), 
                torch.tensor(flat_seq, dtype=torch.long), 
                torch.tensor(flat_types, dtype=torch.long), 
                torch.tensor(target_sid, dtype=torch.long))

    def collate_fn(self, batch):
        # 保持不变
        user_ids, seqs, types, target_sids = zip(*batch)
        B = len(batch)

        user_ids = torch.stack(user_ids)
        sem_ids_fut = torch.stack(target_sids) # 现在这里绝对是 [B, sid_length]

        PAD_TOKEN_ID = self.total_vocab_size
        
        # 组装语义序列 ID
        sem_ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID) 
        seq_mask = (sem_ids != PAD_TOKEN_ID)

        # 组装类型 ID
        token_type_ids = pad_sequence(types, batch_first=True, padding_value=0)
        token_type_ids = token_type_ids.masked_fill(~seq_mask, 0)
        
        token_type_ids_fut = torch.arange(self.sid_length).unsqueeze(0).repeat(B, 1)
        
        return TokenizedSeqBatch(
            user_ids=user_ids, sem_ids=sem_ids, sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask, token_type_ids=token_type_ids, token_type_ids_fut=token_type_ids_fut
        )


class VarlenTigerDatasetSID(Dataset):
    def __init__(self, seqs_file, mapping_file, tiger_code_file, meta_file, max_token_len=64, split='train'):
        print(f"加载变长数据中 ({split} split)...")
        with open(seqs_file, 'r') as f:
            self.all_seqs = json.load(f)
        with open(mapping_file, 'r') as f:
            self.id_mapping = json.load(f)
            
        with open(tiger_code_file, 'r') as f:
            self.item2code = json.load(f)
        
        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        self.user2id = self.id_mapping['user2id']
        self.item2id = self.id_mapping['item2id']
        
        # 唯一容量限制
        self.max_token_len = max_token_len 
        self.sid_length = self.meta["sid_length"] 
        
        vocab_sizes = [
            self.meta[f"vocab_size_layer{i+1}"] for i in range(self.sid_length)
        ]
        
        self.offsets = [0]
        for i in range(len(vocab_sizes) - 1):
            self.offsets.append(self.offsets[-1] + vocab_sizes[i])
            
        self.total_vocab_size = sum(vocab_sizes) 
        
        print(f"[*] 动态加载变长配置成功！最大 SID 长度: {self.sid_length}")
        print(f"[*] 全局总词表大小: {self.total_vocab_size}")
        print(f"[*] 压测模式 -> 仅限制最大 Token 数量: {self.max_token_len}")
        
        self.samples = []
        for user_str, item_str_list in self.all_seqs.items():
            if len(item_str_list) < 4:
                continue
                
            user_int = self.user2id[user_str]
            item_int_str_list = [str(self.item2id[i]) for i in item_str_list]
            
            # 彻底移除了所有的 [-max_seq_len:] 粗截断，保留完整历史序列
            if split == 'test':
                seq_items = item_int_str_list[:-1]
                target_item = item_int_str_list[-1]
                self.samples.append((user_int, seq_items, target_item))
            elif split == 'val':
                seq_items = item_int_str_list[:-2]
                target_item = item_int_str_list[-2]
                self.samples.append((user_int, seq_items, target_item))
            elif split == 'train':
                train_list = item_int_str_list[:-2]
                for i in range(1, len(train_list)):
                    seq_items = train_list[:i]
                    target_item = train_list[i]
                    self.samples.append((user_int, seq_items, target_item))
            else:
                raise ValueError("split 必须是 'train', 'val', 或 'test'")
            
        print(f"[*] 成功构建 {len(self.samples)} 条 {split} 样本。")

    def __len__(self):
        return len(self.samples)
        
    def _get_offset_sid_history(self, item_id_str):
        raw_code = self.item2code[item_id_str]
        actual_len = len(raw_code)
        codes = [raw_code[i] + self.offsets[i] for i in range(actual_len)]
        types = list(range(actual_len))
        return codes, types
    
    def _get_offset_sid_target(self, item_id_str):
        raw_code = self.item2code[item_id_str]
        actual_len = len(raw_code)
        padded_code = [raw_code[i] + self.offsets[i] for i in range(actual_len)]
        pad_token_id = self.total_vocab_size
        while len(padded_code) < self.sid_length:
            padded_code.append(pad_token_id)
            
        return padded_code

    def __getitem__(self, idx):
        user_int, seq_items, target_item = self.samples[idx]
        
        flat_seq = []
        flat_types = []
        
        # 贪婪吞噬逻辑：从最近的交互往回倒推，直到塞满 max_token_len 为止
        # 因为 seq_items 现在是完整的历史，这个 reversed 循环会自动截取最相关的“近期记忆”
        for item in reversed(seq_items):
            codes, types = self._get_offset_sid_history(item)
            
            # 如果加上这个物品的 Token 会超标，说明“胃”已经满了，结束吞噬
            if len(flat_seq) + len(codes) > self.max_token_len:
                break
                
            # 维持正确的时间线顺序
            flat_seq = codes + flat_seq
            flat_types = types + flat_types
            
        target_sid = self._get_offset_sid_target(target_item)
        
        return (torch.tensor(user_int, dtype=torch.long), 
                torch.tensor(flat_seq, dtype=torch.long), 
                torch.tensor(flat_types, dtype=torch.long), 
                torch.tensor(target_sid, dtype=torch.long))

    def collate_fn(self, batch):
        user_ids, seqs, types, target_sids = zip(*batch)
        B = len(batch)

        user_ids = torch.stack(user_ids)
        sem_ids_fut = torch.stack(target_sids) 

        PAD_TOKEN_ID = self.total_vocab_size
        
        sem_ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID) 
        seq_mask = (sem_ids != PAD_TOKEN_ID)

        token_type_ids = pad_sequence(types, batch_first=True, padding_value=0)
        token_type_ids = token_type_ids.masked_fill(~seq_mask, 0)
        
        token_type_ids_fut = torch.arange(self.sid_length).unsqueeze(0).repeat(B, 1)
        
        return TokenizedSeqBatch(
            user_ids=user_ids, sem_ids=sem_ids, sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask, token_type_ids=token_type_ids, token_type_ids_fut=token_type_ids_fut
        )

"""
if __name__ == "__main__":
    # 配置路径 (请根据你服务器的实际路径微调)
    BASE_DIR = "../data/Toys"  # 你的 json 文件夹
    TIGER_DIR = "../rqvae_data/Toys"           # 存放 tiger_item2code 的文件夹
    
    # 实例化测试 Dataset
    test_dataset = TigerDataset(
        seqs_file=f"{BASE_DIR}/all_item_seqs.json",
        mapping_file=f"{BASE_DIR}/id_mapping.json",
        tiger_code_file=f"{TIGER_DIR}/tiger_item2code.json",
        meta_file=f"{TIGER_DIR}/tiger_item2code_meta.json",
        max_seq_len=20,
        split='train'
    )
    
    # 实例化 DataLoader，设置 Batch Size = 2 方便观察
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=test_dataset.collate_fn)
    
    # 抓取第一个 Batch
    batch = next(iter(test_loader))
    
    print("\n" + "="*50)
    print("Batch 数据组装测试报告")
    print("="*50)
    
    print(f"1. User IDs 形态: {batch.user_ids.shape}")
    print(f"   具体数值: {batch.user_ids.tolist()}\n")
    
    print(f"2. 输入历史序列 (sem_ids) 形态: {batch.sem_ids.shape}  <-- 应为 [Batch, N*4]")
    print(f"   样本 0 的前 8 个 token: {batch.sem_ids[0][:8].tolist()}")
    print("   (请检查数值:第1个应<1024, 第2个>1024, 第3个>2048, 第4个>3072)")
    print(f"   是否有 Padding (-1): {'Yes' if -1 in batch.sem_ids else 'No'}\n")
    
    print(f"3. 预测目标 (sem_ids_fut) 形态: {batch.sem_ids_fut.shape} <-- 应为 [Batch, 4]")
    print(f"   具体数值:\n{batch.sem_ids_fut}\n")
    
    print(f"4. 序列掩码 (seq_mask) 形态: {batch.seq_mask.shape}")
    print(f"   样本 0 的尾部情况: {batch.seq_mask[0][-8:].tolist()}\n")
    
    print(f"5. Token 类型 ID (token_type_ids) 形态: {batch.token_type_ids.shape}")
    print(f"   样本 0 的前 8 个类型: {batch.token_type_ids[0][:8].tolist()} <-- 应为 [0, 1, 2, 3, 0, 1, 2, 3]")
    print("="*50)
"""