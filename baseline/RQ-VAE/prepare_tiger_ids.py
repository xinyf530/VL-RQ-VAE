import os
import json
from collections import defaultdict

def generate_tiger_ready_ids(input_json_path, output_json_path):
    print("="*50)
    print("开始构建 TIGER 专用无冲突 SID 字典")
    print("="*50)

    # 1. 加载 RQ-VAE 生成的原始 (有碰撞的) 语义 ID
    with open(input_json_path, 'r', encoding='utf-8') as f:
        item2code = json.load(f)
    
    # 2. 将商品按其前 3 位 Semantic ID 进行分组
    # cluster_map 结构: { (12, 34, 56): ['item_1', 'item_5', 'item_10'] }
    cluster_map = defaultdict(list)
    for item_id, code in item2code.items():
        cluster_map[tuple(code)].append(item_id)
        
    print(f"统计: 共有 {len(item2code)} 个商品，形成了 {len(cluster_map)} 个独特的语义簇(Clusters)。")

    # 3. 分配冲突解决位 (Tie-breaker Token)
    tiger_item2code = {}
    max_cluster_size = 0
    
    for code_tuple, item_list in cluster_map.items():
        # 记录最大的碰撞簇，这决定了下游 TIGER 第 4 层的 Vocabulary Size
        if len(item_list) > max_cluster_size:
            max_cluster_size = len(item_list)
            
        # 给簇内的每一个商品分配一个局部的 Unique ID (从 0 开始递增)
        # 例如: item_1 变成 [12, 34, 56, 0]
        #       item_5 变成 [12, 34, 56, 1]
        for local_id, item_id in enumerate(sorted(item_list)):
            tiger_code = list(code_tuple) + [local_id]
            tiger_item2code[item_id] = tiger_code
            
    # 4. 保存为 TIGER 可直接使用的字典
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(tiger_item2code, f)

    # 5. 生成并保存 meta 信息，供下游 TIGER 模型自动读取配置
    # vocab_size_layer4 固定为 256，作为安全上界，不随数据集变化
    FIXED_VOCAB_SIZE_LAYER4 = 256
    meta = {
        "vocab_size_layer1": 1024,
        "vocab_size_layer2": 1024,
        "vocab_size_layer3": 1024,
        "vocab_size_layer4": FIXED_VOCAB_SIZE_LAYER4,
        "actual_max_cluster": max_cluster_size,
        "total_items": len(item2code),
        "total_unique_sids": len(cluster_map),
        "sid_length": 4
    }
    meta_output_path = output_json_path.replace(".json", "_meta.json")
    with open(meta_output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
        
    print("\n转换完成！")
    print(f"  - 输入路径: {input_json_path}")
    print(f"  - 输出路径: {output_json_path}")
    print(f"  - Meta 路径: {meta_output_path}")
    print(f"  - 总物品数: {meta['total_items']}")
    print(f"  - 实际最大碰撞簇大小: {max_cluster_size}")
    print(f"  - 第 4 层固定 vocab_size: {FIXED_VOCAB_SIZE_LAYER4}  (安全上界，不随数据集变化)")
    print(f"  - 下游 TIGER 可直接读取 {meta_output_path} 获取模型配置")
    print("="*50)

    return meta

if __name__ == "__main__":
    BASE_DIR = "rqvae_data/ele"
    INPUT_JSON = os.path.join(BASE_DIR, "item2code_baseline.json")
    
    # 输出一个新的文件，专供 TIGER 训练使用
    OUTPUT_JSON = os.path.join(BASE_DIR, "item2code_final.json")
    
    meta = generate_tiger_ready_ids(INPUT_JSON, OUTPUT_JSON)
    print("\nMeta 信息预览:")
    print(json.dumps(meta, indent=2))
