
import os
import json
import re
from collections import defaultdict

def clean_text(raw_text):
    if not raw_text: return ""
    if isinstance(raw_text, list): raw_text = ' '.join(str(item) for item in raw_text)
    text = str(raw_text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith(('.', '!', '?')): text += '.'
    return text

def process_domain_unidirectional(domain_name, review_file, meta_file, base_cache_dir="cache"):
    print(f"\n{'='*20} 开始处理 {domain_name} 域 (单向长尾保留策略) {'='*20}")
    
    # --- 核心配置 ---
    # 2018-01-01 00:00:00 1514764800000
    # 2022-01-01 00:00:00 1640995200000
    MIN_TIMESTAMP = 1514764800000  # >= 2014-01-01  1388534400000
    MIN_RATING = 3.0               # 保留 >= 3.0 的中性及正向反馈
    MIN_INTERACTIONS = 5           # 仅限制用户侧 5-core
    
    # ==========================================
    # 步骤 1: 读取数据，执行时间和评分过滤
    # ==========================================
    user_interactions = defaultdict(list)
    print(f"1. 读取 Review，过滤时间(>=2018)与评分(>={MIN_RATING})...")
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            item_id = data.get('parent_asin')
            timestamp = int(data.get('timestamp', 0))
            rating = float(data.get('rating', 0.0))
            
            if timestamp >= MIN_TIMESTAMP and rating >= MIN_RATING:
                user_interactions[user_id].append((item_id, timestamp))

    # ==========================================
    # 步骤 2: 单向 5-core 过滤 (仅过滤低频用户，全量保留物品)
    # ==========================================
    print(f"2. 开始执行单向 {MIN_INTERACTIONS}-core 过滤 (保留长尾物品)...")
    filtered_users = {}
    valid_items = set()
    
    for user, seq in user_interactions.items():
        if len(seq) >= MIN_INTERACTIONS:
            filtered_users[user] = seq
            for item, _ in seq:
                valid_items.add(item)  # 哪怕物品只出现 1 次，只要被有效用户交互过就保留
                
    user_interactions = filtered_users

    # ==========================================
    # 步骤 3: 排序与字典构建
    # ==========================================
    print("3. 按时间排序生成历史序列并构建 ID Mapping...")
    all_item_seqs = {}
    user2id, item2id = {}, {}
    id2user, id2item = ["<pad>"], ["<pad>"]
    
    for user, seq in user_interactions.items():
        seq.sort(key=lambda x: x[1])
        all_item_seqs[user] = [item for item, _ in seq]
        user2id[user] = len(id2user)
        id2user.append(user)
        
    for item in sorted(list(valid_items)):
        item2id[item] = len(id2item)
        id2item.append(item)

    id_mapping = {'user2id': user2id, 'item2id': item2id, 'id2user': id2user, 'id2item': id2item}

    # ==========================================
    # 步骤 4: 提取丰富的 Metadata (精准对齐 BGE 需求)
    # ==========================================
    print("4. 读取 Metadata 提取目标字段...")
    item2meta = {}
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            item_id = data.get('parent_asin', data.get('asin'))
            
            if item_id in item2id:
                features_list = []
                if data.get('title'): features_list.append("Title: " + clean_text(data['title']))
                if data.get('main_category'): features_list.append("Main Category: " + clean_text(data['main_category']))
                # 3. Categories (平级分类标签拼接)
                if data.get('categories') and isinstance(data['categories'], list):
                    # 使用逗号进行并列拼接，避免误导语义模型
                    cat_str = ", ".join([clean_text(c) for c in data['categories']])
                    features_list.append("Categories: " + cat_str)
                if data.get('brand'): features_list.append("Brand: " + clean_text(data['brand']))
                
                price_val = data.get('price')
                if price_val is not None:
                    try:
                        if isinstance(price_val, str):
                            cleaned_price = re.sub(r'[^\d.-]', '', price_val.strip())
                            if cleaned_price: price_val = float(cleaned_price)
                            else: price_val = None
                        else: price_val = float(price_val)
                        if price_val is not None: features_list.append(f"Price: {price_val:.2f}")
                    except: pass

                avg_rating = data.get('average_rating')
                if avg_rating is not None:
                    try: features_list.append(f"Average Rating: {float(avg_rating):.1f}")
                    except: pass
                
                if data.get('features'):
                    raw_feat = data['features']
                    if isinstance(raw_feat, list): raw_feat = ' '.join(raw_feat)
                    features_list.append("Features: " + clean_text(raw_feat))
                
                sentence = ' '.join(features_list).strip()
                item2meta[item_id] = sentence if sentence else "No description available."

    for item in valid_items:
        if item not in item2meta: item2meta[item] = "No description available."

    # ==========================================
    # 步骤 5: 存储结果并打印统计
    # ==========================================
    processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', domain_name, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    with open(os.path.join(processed_dir, 'all_item_seqs.json'), 'w') as f: json.dump(all_item_seqs, f)
    with open(os.path.join(processed_dir, 'id_mapping.json'), 'w') as f: json.dump(id_mapping, f)
    with open(os.path.join(processed_dir, 'metadata.sentence.json'), 'w', encoding='utf-8') as f: 
        json.dump(item2meta, f, ensure_ascii=False, indent=4)
        
    total_users = len(user2id)
    total_items = len(item2id)
    total_interactions = sum(len(seq) for seq in all_item_seqs.values())
    sparsity = 1.0 - (total_interactions / (total_users * total_items))
    avg_interaction_length = total_interactions / total_users if total_users > 0 else 0
    
    print(f"[*] {domain_name} 域单向过滤处理完毕！")
    print(f"    - 保留用户数: {total_users}")
    print(f"    - 保留物品数: {total_items} (海量长尾已保留！)")
    print(f"    - 剩余交互数: {total_interactions}")
    print(f"    - 平均交互长度: {avg_interaction_length:.2f} (次/用户)")
    print(f"    - 数据稀疏度: {sparsity:.6f}")

if __name__ == "__main__":
    DATA_DIR1 = "/data1/datasets/amazon2023/review"
    DATA_DIR2 = "/data1/datasets/amazon2023/meta"
    CACHE_DIR = "/data1/xinyuefeng/VL-RQ-VAE/baseline/RQ-VAE/data/Toys"
    
    domains_to_process = {"Toys": {"review": os.path.join(DATA_DIR1, "Toys_and_Games.jsonl"), "meta": os.path.join(DATA_DIR2, "meta_Toys_and_Games.jsonl")}}
    for domain_name, paths in domains_to_process.items():
        if os.path.exists(paths["review"]) and os.path.exists(paths["meta"]):
            process_domain_unidirectional(domain_name, paths["review"], paths["meta"], CACHE_DIR)
"""

import os
import json
import re
from collections import defaultdict

def clean_text(raw_text):
    if not raw_text: return ""
    if isinstance(raw_text, list): raw_text = ' '.join(str(item) for item in raw_text)
    text = str(raw_text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith(('.', '!', '?')): text += '.'
    return text

def process_domain_unidirectional(domain_name, review_file, meta_file, base_cache_dir="cache"):
    print(f"\n{'='*20} 开始处理 {domain_name} 域 (双向5-core过滤策略) {'='*20}")
    
    # --- 核心配置 ---
    # 2018-01-01 00:00:00 1514764800000
    # 2022-01-01 00:00:00 1640995200000
    # 2020-01-01 00:00:00 1577836800000
    MIN_TIMESTAMP = 1577836800000  # >= 2014-01-01  1388534400000
    MIN_RATING = 3.0               # 保留 >= 3.0 的中性及正向反馈
    MIN_INTERACTIONS = 5           # 仅限制用户侧 5-core
    
    # ==========================================
    # 步骤 1: 读取数据，执行时间和评分过滤
    # ==========================================
    user_interactions = defaultdict(list)
    print(f"1. 读取 Review，过滤时间(>=2018)与评分(>={MIN_RATING})...")
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            item_id = data.get('parent_asin')
            timestamp = int(data.get('timestamp', 0))
            rating = float(data.get('rating', 0.0))
            
            if timestamp >= MIN_TIMESTAMP and rating >= MIN_RATING:
                user_interactions[user_id].append((item_id, timestamp))

    # ==========================================
    # 步骤 2: 双向 5-core 迭代过滤 (交替移除低频用户和低频物品，直到收敛)
    # ==========================================
    print(f"2. 开始执行双向 {MIN_INTERACTIONS}-core 迭代过滤...")
    iteration = 0
    while True:
        iteration += 1
        prev_user_count = len(user_interactions)
        
        # 2a. User-side: remove users with < MIN_INTERACTIONS
        user_interactions = {
            user: seq for user, seq in user_interactions.items()
            if len(seq) >= MIN_INTERACTIONS
        }
        
        # 2b. Item-side: count item frequency across all remaining users
        item_count = defaultdict(int)
        for seq in user_interactions.values():
            for item, _ in seq:
                item_count[item] += 1
        valid_items = {item for item, cnt in item_count.items() if cnt >= MIN_INTERACTIONS}
        
        # 2c. Remove interactions with low-frequency items from user sequences
        user_interactions = {
            user: [(item, ts) for item, ts in seq if item in valid_items]
            for user, seq in user_interactions.items()
        }
        # Remove users that became too short after item filtering
        user_interactions = {
            user: seq for user, seq in user_interactions.items()
            if len(seq) >= MIN_INTERACTIONS
        }
        
        curr_user_count = len(user_interactions)
        curr_item_count = len(valid_items)
        print(f"   迭代 {iteration}: 用户 {curr_user_count}, 物品 {curr_item_count}")
        
        # Converged: no users were removed in this round
        if curr_user_count == prev_user_count:
            break
    
    print(f"   双向 {MIN_INTERACTIONS}-core 过滤收敛，共迭代 {iteration} 轮")

    # ==========================================
    # 步骤 3: 排序与字典构建
    # ==========================================
    print("3. 按时间排序生成历史序列并构建 ID Mapping...")
    all_item_seqs = {}
    user2id, item2id = {}, {}
    id2user, id2item = ["<pad>"], ["<pad>"]
    
    for user, seq in user_interactions.items():
        seq.sort(key=lambda x: x[1])
        all_item_seqs[user] = [item for item, _ in seq]
        user2id[user] = len(id2user)
        id2user.append(user)
        
    for item in sorted(list(valid_items)):
        item2id[item] = len(id2item)
        id2item.append(item)

    id_mapping = {'user2id': user2id, 'item2id': item2id, 'id2user': id2user, 'id2item': id2item}

    # ==========================================
    # 步骤 4: 提取丰富的 Metadata (精准对齐 BGE 需求)
    # ==========================================
    print("4. 读取 Metadata 提取目标字段...")
    item2meta = {}
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            item_id = data.get('parent_asin', data.get('asin'))
            
            if item_id in item2id:
                features_list = []
                if data.get('title'): features_list.append("Title: " + clean_text(data['title']))
                if data.get('main_category'): features_list.append("Main Category: " + clean_text(data['main_category']))
                # 3. Categories (平级分类标签拼接)
                if data.get('categories') and isinstance(data['categories'], list):
                    # 使用逗号进行并列拼接，避免误导语义模型
                    cat_str = ", ".join([clean_text(c) for c in data['categories']])
                    features_list.append("Categories: " + cat_str)
                if data.get('brand'): features_list.append("Brand: " + clean_text(data['brand']))
                
                price_val = data.get('price')
                if price_val is not None:
                    try:
                        if isinstance(price_val, str):
                            cleaned_price = re.sub(r'[^\d.-]', '', price_val.strip())
                            if cleaned_price: price_val = float(cleaned_price)
                            else: price_val = None
                        else: price_val = float(price_val)
                        if price_val is not None: features_list.append(f"Price: {price_val:.2f}")
                    except: pass

                avg_rating = data.get('average_rating')
                if avg_rating is not None:
                    try: features_list.append(f"Average Rating: {float(avg_rating):.1f}")
                    except: pass
                
                if data.get('features'):
                    raw_feat = data['features']
                    if isinstance(raw_feat, list): raw_feat = ' '.join(raw_feat)
                    features_list.append("Features: " + clean_text(raw_feat))
                
                sentence = ' '.join(features_list).strip()
                item2meta[item_id] = sentence if sentence else "No description available."

    for item in valid_items:
        if item not in item2meta: item2meta[item] = "No description available."

    # ==========================================
    # 步骤 5: 存储结果并打印统计
    # ==========================================
    processed_dir = os.path.join(base_cache_dir, 'AmazonReviews2014', domain_name, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    with open(os.path.join(processed_dir, 'all_item_seqs.json'), 'w') as f: json.dump(all_item_seqs, f)
    with open(os.path.join(processed_dir, 'id_mapping.json'), 'w') as f: json.dump(id_mapping, f)
    with open(os.path.join(processed_dir, 'metadata.sentence.json'), 'w', encoding='utf-8') as f: 
        json.dump(item2meta, f, ensure_ascii=False, indent=4)
        
    total_users = len(user2id)
    total_items = len(item2id)
    total_interactions = sum(len(seq) for seq in all_item_seqs.values())
    sparsity = 1.0 - (total_interactions / (total_users * total_items))
    avg_interaction_length = total_interactions / total_users if total_users > 0 else 0
    
    print(f"[*] {domain_name} 域双向5-core过滤处理完毕！")
    print(f"    - 保留用户数: {total_users}")
    print(f"    - 保留物品数: {total_items} (海量长尾已保留！)")
    print(f"    - 剩余交互数: {total_interactions}")
    print(f"    - 平均交互长度: {avg_interaction_length:.2f} (次/用户)")
    print(f"    - 数据稀疏度: {sparsity:.6f}")

if __name__ == "__main__":
    DATA_DIR1 = "/workspace/my_folder/luozijian/dataset/amazon2023/review"
    DATA_DIR2 = "/workspace/my_folder/luozijian/dataset/amazon2023/meta"
    CACHE_DIR = "/workspace/user_code/baseline/RQ-VAE/data/ele"
    
    #domains_to_process = {"phone": {"review": os.path.join(DATA_DIR1, "Cell_Phones_and_Accessories.jsonl"), "meta": os.path.join(DATA_DIR2, "meta_Cell_Phones_and_Accessories.jsonl")}}
    domains_to_process = {"ele": {"review": os.path.join(DATA_DIR1, "Electronics.jsonl"), "meta": os.path.join(DATA_DIR2, "meta_Electronics.jsonl")}}
    for domain_name, paths in domains_to_process.items():
        if os.path.exists(paths["review"]) and os.path.exists(paths["meta"]):
            process_domain_unidirectional(domain_name, paths["review"], paths["meta"], CACHE_DIR)

"""