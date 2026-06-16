import json
import os

# 把你想检查的文件名写在这里
files_to_check = [
    "/workspace/my_folder/luozijian/dataset/amazon2023/review/Toys_and_Games.jsonl",
    "/workspace/my_folder/luozijian/dataset/amazon2023/meta/meta_Toys_and_Games.jsonl"
]

def preview_jsonl(file_path, num_lines=2):
    """只读取并打印 jsonl 文件的前 num_lines 行"""
    print(f"\n{'='*40}")
    print(f"👀 正在预览: {file_path}")
    print(f"{'='*40}")
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path} (请确认是否已解压完毕)")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                # 解析单行 JSON 并格式化输出
                data = json.loads(line.strip())
                print(f"\n🔹 第 {i+1} 条数据:")
                print(json.dumps(data, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️ 读取时发生错误: {e}")

if __name__ == "__main__":
    for file in files_to_check:
        preview_jsonl(file, num_lines=2) # 默认看前2条数据