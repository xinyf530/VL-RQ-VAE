import torch
import os
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_train_val_loaders(emb_path, batch_size=2048, val_ratio=0.05):
    """
    加载嵌入并划分为训练集和验证集
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"未找到嵌入文件: {emb_path}")
        
    embeddings = torch.load(emb_path) # Shape: [N, 1024]
    dataset = TensorDataset(embeddings)
    
    # 计算划分数量
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # 随机划分
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # 固定随机种子以便复现
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, len(embeddings)

def get_export_loader(emb_path, batch_size=2048):
    """
    获取全量物品的顺序加载器，用于最后导出 Semantic IDs
    """
    embeddings = torch.load(emb_path)
    dataset = TensorDataset(embeddings)
    # 必须 shuffle=False 以保证索引与 Item ID 的映射关系
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)