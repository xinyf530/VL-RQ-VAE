import os
import json
import torch
from sentence_transformers import SentenceTransformer

def generate_full_embeddings(meta_json_path, id_mapping_path, output_pt_path, batch_size=64):
    print("1. 加载 BGE-large-en 模型...")
    # 指定你需要的 GPU 设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = SentenceTransformer('/workspace/my_folder/luozijian/models/bge-en', device=device)
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    
    # 开启半精度推理以节省显存并加速
    if device == "cuda:3":
        model.half()

    print(f"2. 读取 Metadata 和 ID Mapping...")
    with open(meta_json_path, 'r', encoding='utf-8') as f:
        item2meta = json.load(f)
        
    with open(id_mapping_path, 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
        
    # 获取预处理时生成的严格映射表
    id2item = id_mapping['id2item']
    
    # 【核心逻辑】：跳过索引 0 的 "<pad>"，保留真实的商品 ASIN 列表
    # 此时 real_items[0] 对应预处理中的 item_id 1，real_items[1] 对应 item_id 2 ...
    real_items = id2item[1:]
    
    # 严格按照映射后的 item_id 顺序提取文本
    texts = [item2meta.get(item_asin, "No description available.") for item_asin in real_items]
    
    total_items = len(texts)
    print(f"   共加载 {total_items} 条真实商品文本。")
    print(f"3. 开始批量提取 1024 维原生特征 (Batch Size: {batch_size})...")
    
    # 提取向量 (直接返回 PyTorch Tensor)
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True, # 保证向量在单位超球面上，极度利于 RQ-VAE 聚类
        convert_to_tensor=True     # 直接输出 Tensor，无需 numpy 中转
    )
    
    # 将 Tensor 移至 CPU 并确保是 float32 类型，方便后续保存和读取
    embeddings = embeddings.cpu().float()
    
    print(f"4. 保存 Tensor 至 .pt 文件...")
    # 保存为 PyTorch 原生格式
    torch.save(embeddings, output_pt_path)
    
    print("\n" + "="*50)
    print(f"✅ 提取成功！")
    print(f"   - 矩阵维度: {embeddings.shape} (预期应为 [{total_items}, 1024])")
    print(f"   - 💡 偏移法则已确认: 矩阵的 row 0 将绝对对应你的 item_id 1！")
    print(f"   - 文件大小: {os.path.getsize(output_pt_path) / (1024*1024):.2f} MB")
    print(f"   - 保存路径: {output_pt_path}")
    print("="*50)

if __name__ == "__main__":
    BASE_DIR = "/data1/xinyuefeng/VL-RQ-VAE/baseline/RQ-VAE/data/Toys/AmazonReviews2014/Toys/processed"
    META_PATH = os.path.join(BASE_DIR, "metadata.sentence.json")
    MAP_PATH = os.path.join(BASE_DIR, "id_mapping.json")
    OUTPUT_PATH = os.path.join(BASE_DIR, "bge_embeddings.pt")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    generate_full_embeddings(META_PATH, MAP_PATH, OUTPUT_PATH, batch_size=64)