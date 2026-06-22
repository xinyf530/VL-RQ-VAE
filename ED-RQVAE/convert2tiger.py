import os
import json
from collections import defaultdict

def main():
    # ==== 路径配置 ====
    # 替换为你 ED-RQVAE 实际生成的 json 路径
    INPUT_JSON = "edrqvae_data/Toys/1.0/item2code_final.json" 
    
    OUTPUT_JSON = "tiger_data/Toys/tiger_item2code.json"
    OUTPUT_LENGTH = "tiger_data/Toys/tiger_item2length.json"
    OUTPUT_META = "tiger_data/Toys/tiger_item2code_meta.json"

    VARLEN_VOCAB_SIZE = 1024
    UNIQUE_OFFSET = VARLEN_VOCAB_SIZE 

    print(f"读取原始变长字典 {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        sids_raw = json.load(f)
        
    # 🌟 1. 动态获取数据中真实的全局最大有效长度
    # 遍历所有的 code 数组，找出最长的一个
    MAX_ACTUAL_LENGTH = max(len(code) for code in sids_raw.values())
    TOTAL_SID_LENGTH = MAX_ACTUAL_LENGTH + 1  # 总长度 = 真实最大变长 + 1个 Exact_ID
    
    print(f"检测到全局最大有效语义长度为: {MAX_ACTUAL_LENGTH}")
    print(f"加上 Exact_ID 后，Tiger 输入的全局定长将对齐为: {TOTAL_SID_LENGTH}")

    # 2. 将商品按变长前缀分组 (找出物理克隆体)
    cluster_map = defaultdict(list)
    for item_id, code in sids_raw.items():
        valid_sid = tuple(code)
        cluster_map[valid_sid].append(item_id)

    max_cluster_size = max(len(items) for items in cluster_map.values())
    
    # 计算有效的语义 + UniqueID 词汇量
    effective_vocab = max(VARLEN_VOCAB_SIZE, UNIQUE_OFFSET + max_cluster_size)
    if effective_vocab <= 1024: effective_vocab = 1024
    elif effective_vocab <= 2048: effective_vocab = 2048
    else: effective_vocab = 2 ** (effective_vocab - 1).bit_length()

    # 🌟 绝杀：使用 effective_vocab 作为专用的 LAYER_PAD_ID
    LAYER_PAD_ID = effective_vocab

    tiger_item2code = {}
    tiger_item2length = {}

    print("开始进行 Exact_ID 分配与 PAD 补齐...")
    for valid_sid, items in cluster_map.items():
        # 对同一簇内的 item_id 排序，保证每次生成的 Exact_ID 是固定的
        items_sorted = sorted(items, key=lambda x: int(x) if str(x).isdigit() else x)
        
        for uid, item_id in enumerate(items_sorted):
            orig_length = len(valid_sid)

            code = list(valid_sid)                                  # 1. 填入完整保留的变长前缀
            code.append(UNIQUE_OFFSET + uid)                        # 2. 紧跟 Exact_ID 区分克隆体
            
            # 🌟 3. 不足 TOTAL_SID_LENGTH 的部分用 LAYER_PAD_ID 补齐
            code += [LAYER_PAD_ID] * (TOTAL_SID_LENGTH - len(code))            

            tiger_item2code[str(item_id)] = code
            tiger_item2length[str(item_id)] = orig_length + 1 

    # 🌟 每层的词表大小现在是 effective_vocab + 1 (包含了专门的 LAYER_PAD_ID)
    layer_vocab_size = effective_vocab + 1

    # 3. 保存输出
    print("保存 Tiger 格式字典...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f: 
        json.dump(tiger_item2code, f)
    with open(OUTPUT_LENGTH, "w", encoding="utf-8") as f: 
        json.dump(tiger_item2length, f)
    
    meta = {
        **{f"vocab_size_layer{i + 1}": layer_vocab_size for i in range(TOTAL_SID_LENGTH)},
        "effective_vocab_per_layer": layer_vocab_size,
        "actual_max_cluster": max_cluster_size,
        "max_actual_length": MAX_ACTUAL_LENGTH,
        "sid_length": TOTAL_SID_LENGTH,
        "pad_id": LAYER_PAD_ID
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f: 
        json.dump(meta, f, indent=2)

    print("转换大功告成！")
    print(f"  - Tiger 序列输入长度 (sid_length): {TOTAL_SID_LENGTH}")
    print(f"  - 专属填充码 PAD_ID: {LAYER_PAD_ID}")
    print(f"  - 统一层级词表大小: {layer_vocab_size}")

if __name__ == "__main__":
    main()