import json
import numpy as np

# 加载数据
with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/all_item_seqs.json', 'r') as f:
    all_sequences = json.load(f)
with open('/workspace/user_code/baseline/RQ-VAE/data/Toys/id_mapping.json', 'r') as f:
    mapping = json.load(f)
    item2id = mapping['item2id']

# 1. 统计训练集中的物品频次 (Leave-One-Out: 序列前 n-2 个)
train_counts = {}
all_items_in_catalog = set(item2id.keys())

for user_id, seq in all_sequences.items():
    # 留一法下，最后两个分别是 Test 和 Val，其余是 Train
    train_seq = seq[:-2] 
    for item in train_seq:
        train_counts[item] = train_counts.get(item, 0) + 1

# 2. 识别 Unseen 物品
train_items = set(train_counts.keys())
unseen_items = all_items_in_catalog - train_items
unseen_count = len(unseen_items)

# 3. 对训练集物品按流行度排序，划分 Hot 和 Tail
# 仅针对在训练集中出现过的物品进行帕累托划分
sorted_train_items = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
num_train_items = len(sorted_train_items)
hot_cutoff_idx = int(0.2 * num_train_items)

hot_items_list = sorted_train_items[:hot_cutoff_idx]
tail_items_list = sorted_train_items[hot_cutoff_idx:]

# 4. 获取阈值和数量
hot_threshold = hot_items_list[-1][1] if hot_items_list else 0
print(f"统计结果:")
print(f"总物品数 (Catalog Size): {len(all_items_in_catalog)}")
print(f"Unseen 物品数 (频次=0): {unseen_count} ({unseen_count/len(all_items_in_catalog):.2%})")
print(f"训练集物品数 (频次>0): {num_train_items}")
print(f"--- 基于训练集物品的二八划分 ---")
print(f"Hot 物品数 (前20%): {len(hot_items_list)} (最小频次: {hot_threshold})")
print(f"Tail/Cold 物品数 (后80%): {len(tail_items_list)}")