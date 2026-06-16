import json
from collections import Counter

def count_infrequent_items(seq_file, mapping_file, threshold=5):
    print("正在加载数据，请稍候...")
    
    # 1. 读取映射文件，获取所有已知的物品列表
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
            item2id = id_mapping.get('item2id', {})
            all_known_items = set(item2id.keys())
    except FileNotFoundError:
        print(f"错误: 找不到文件 {mapping_file}")
        return

    # 2. 读取交互序列文件
    try:
        with open(seq_file, 'r', encoding='utf-8') as f:
            all_item_seqs = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {seq_file}")
        return

    # 3. 统计每个物品的交互次数
    item_counts = Counter()
    for user, items in all_item_seqs.items():
        item_counts.update(items)

    # 4. 筛选交互次数少于 threshold (默认5次) 的物品
    infrequent_items_count = 0
    
    # 我们遍历所有的已知物品，这样可以包含那些在 seq_file 中交互次数为 0 的物品
    for item in all_known_items:
        if item_counts[item] < threshold:
            infrequent_items_count += 1
            
    # 如果你想把序列文件里出现过、但不在映射文件里的物品也算上（作为容错处理）
    # 可以额外检查 item_counts 里的 item 是否在 all_known_items 中
    for item in item_counts:
        if item not in all_known_items and item_counts[item] < threshold:
            infrequent_items_count += 1

    # 5. 输出结果
    print("-" * 30)
    print(f"物品总数 (基于 item2id): {len(all_known_items)}")
    print(f"有交互记录的物品数: {len(item_counts)}")
    print(f"交互次数少于 {threshold} 次的物品数量: {infrequent_items_count}")
    print("-" * 30)

if __name__ == "__main__":
    # 请确保这两个文件与脚本在同一个目录下，或者替换为绝对路径
    SEQ_FILE_PATH = '/workspace/user_code/baseline/RQ-VAE/data/Toys/all_item_seqs.json'
    MAPPING_FILE_PATH = '/workspace/user_code/baseline/RQ-VAE/data/Toys/id_mapping.json'
    
    count_infrequent_items(SEQ_FILE_PATH, MAPPING_FILE_PATH, threshold=5)