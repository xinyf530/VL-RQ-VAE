import json

def generate_dynamic_meta(json_path, output_meta_path):
    print(f"正在读取变长码本文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        item2code = json.load(f)

    # 1. 找出全局最大深度
    max_length = max((len(code) for code in item2code.values() if code), default=0)

    # 2. 初始化一个数组，用于记录【每一层】出现的最大 Token ID
    max_id_per_layer = [-1] * max_length

    # 3. 逐层、逐个商品扫描，寻找真实的 Token ID 天花板
    for code in item2code.values():
        for i, token_id in enumerate(code):
            if token_id > max_id_per_layer[i]:
                max_id_per_layer[i] = token_id

    print("\n" + "="*50)
    print("逐层极致压缩扫描完成！")
    print(f"最大 SID 深度: {max_length} 层")
    print("="*50)

    # 4. 构建精准的 Meta 字典
    meta = {
        "sid_length": max_length,
        "max_actual_length": max_length
    }

    total_vocab = 0
    for i in range(max_length):
        # 该层绝对安全且最小的词表容量 = 最大 ID + 1 (因为 ID 是从 0 开始的)
        layer_vocab_size = max_id_per_layer[i] + 1
        meta[f"vocab_size_layer{i+1}"] = layer_vocab_size
        total_vocab += layer_vocab_size
        
        print(f" ↳ Layer {i+1} 真实最大 ID: {max_id_per_layer[i]:<4} -> 分配 Vocab Size: {layer_vocab_size}")

    meta["total_vocab_size"] = total_vocab
    print("-" * 50)
    print(f"优化后的全局总词表大小: {total_vocab} (如果不优化则是 {max_length * 1024})")
    print(f"为 Tiger 模型省下了 {(max_length * 1024 - total_vocab) * 100 / (max_length * 1024):.2f}% 的无效 Embedding 坑位！")

    # 5. 保存输出
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)
        
    print(f"\n[*] 极致压缩版 Meta 已保存至: {output_meta_path}")

if __name__ == "__main__":
    # 请根据你的实际路径修改
    input_json = "/workspace/user_code/ED-RQVAE/edrqvae_data/Toys/v2/item2code_final.json"
    output_meta = "/workspace/user_code/ED-RQVAE/tiger_data/Toys/edrqvae/tiger_item2code_meta.json"
    
    generate_dynamic_meta(input_json, output_meta)