import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def analyze_token_lengths(meta_json_path, model_name='BAAI/bge-large-en-v1.5', max_limit=512):
    print(f"1. 正在加载 {model_name} 的 Tokenizer...")
    # 加载 BGE 原生分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"2. 正在读取文件: {meta_json_path}")
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        item2meta = json.load(f)
        
    texts = list(item2meta.values())
    total_items = len(texts)
    print(f"   共加载 {total_items} 条商品文本。")
    
    print("3. 开始进行真实 Token 切分计算 (可能需要几分钟)...")
    token_lengths = []
    
    # 遍历计算每条文本的 Token 数量
    for text in tqdm(texts, desc="Tokenizing"):
        # truncation=False 保证我们能拿到文本真实的完整长度
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
        token_lengths.append(len(tokens))
        
    # 转换为 NumPy 数组方便统计
    lengths_arr = np.array(token_lengths)
    
    # ================= 统计指标 =================
    avg_len = np.mean(lengths_arr)
    max_len = np.max(lengths_arr)
    min_len = np.min(lengths_arr)
    p90 = np.percentile(lengths_arr, 90)
    p95 = np.percentile(lengths_arr, 95)
    p99 = np.percentile(lengths_arr, 99)
    
    # 统计超限情况
    exceed_count = np.sum(lengths_arr > max_limit)
    exceed_ratio = (exceed_count / total_items) * 100
    
    print("\n" + "="*40)
    print("📊 BGE Token 长度分布分析报告")
    print("="*40)
    print(f"模型限制 (Max Limit): {max_limit} Tokens")
    print("-" * 40)
    print(f"最短 Token 数: {min_len}")
    print(f"最长 Token 数: {max_len}")
    print(f"平均 Token 数: {avg_len:.1f}")
    print("-" * 40)
    print(f"90% 的文本长度 <= {p90:.0f} Tokens")
    print(f"95% 的文本长度 <= {p95:.0f} Tokens")
    print(f"99% 的文本长度 <= {p99:.0f} Tokens")
    print("-" * 40)
    
    if exceed_count > 0:
        print(f"⚠️ 警告: 共有 {exceed_count} 条文本 ({exceed_ratio:.2f}%) 超出了 {max_limit} 的限制！")
        print("建议: 如果超限比例极低 (如 <1%)，可以直接忽略，模型会自动截断。")
        print("      如果超限比例较高，可能需要回退并限制 Features 字段的最大字符数。")
    else:
        print(f"✅ 完美！没有任何文本超出 {max_limit} Tokens 的限制！")

if __name__ == "__main__":
    META_PATH = "/workspace/user_code/baseline/RQ-VAE/data/Phone/metadata.sentence.json"
    
    # 请确保环境中安装了 transformers 库: pip install transformers
    analyze_token_lengths(META_PATH, model_name='/workspace/my_folder/luozijian/models/bge-en')