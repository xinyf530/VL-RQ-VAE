import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_and_plot_long_tail(seqs_path, output_img_path="long_tail_distribution.png"):
    print(f"1. 正在加载交互序列数据: {seqs_path} ...")
    with open(seqs_path, 'r', encoding='utf-8') as f:
        all_item_seqs = json.load(f)
        
    print("2. 正在统计商品交互频次...")
    # 统计所有出现过的 item 频次
    item_counts = Counter()
    for user, seq in all_item_seqs.items():
        item_counts.update(seq)
        
    # 按频次从大到小排序
    sorted_counts = sorted(item_counts.values(), reverse=True)
    total_items = len(sorted_counts)
    total_interactions = sum(sorted_counts)
    
    # 转换为 numpy array 方便计算
    counts_array = np.array(sorted_counts)
    
    # ==========================================
    # 统计方案 A：基于物品数量的 20/80 划分 (Top 20% Items)
    # ==========================================
    head_item_cutoff = int(total_items * 0.2)
    head_items_A = head_item_cutoff
    tail_items_A = total_items - head_item_cutoff
    head_interactions_A = np.sum(counts_array[:head_item_cutoff])
    tail_interactions_A = total_interactions - head_interactions_A
    
    # ==========================================
    # 统计方案 B：基于交互总量的 80/20 划分 (Top 80% Interactions)
    # ==========================================
    cumulative_interactions = np.cumsum(counts_array)
    # 找到累计交互量刚超过 80% 的索引
    head_cutoff_idx_B = np.argmax(cumulative_interactions >= total_interactions * 0.8) + 1
    head_items_B = head_cutoff_idx_B
    tail_items_B = total_items - head_items_B
    
    # ==========================================
    # 终端打印极其震撼的统计报告
    # ==========================================
    print("\n" + "="*50)
    print("📊 推荐系统长尾分布 (Long-Tail Distribution) 报告")
    print("="*50)
    print(f"总物品数 (Items): {total_items:,}")
    print(f"总交互数 (Interactions): {total_interactions:,}")
    print("-" * 50)
    print("【划分标准 A：Top 20% 物品作为头部】")
    print(f"  - 头部物品 (Head): {head_items_A:,} 个 (20.00%) -> 贡献了 {head_interactions_A/total_interactions*100:.2f}% 的交互")
    print(f"  - 尾部物品 (Tail): {tail_items_A:,} 个 (80.00%) -> 仅贡献了 {tail_interactions_A/total_interactions*100:.2f}% 的交互")
    print("-" * 50)
    print("【划分标准 B：贡献 80% 交互的物品作为头部】")
    print(f"  - 头部物品 (Head): {head_items_B:,} 个 ({head_items_B/total_items*100:.2f}%) -> 贡献了 80.00% 的交互")
    print(f"  - 尾部物品 (Tail): {tail_items_B:,} 个 ({tail_items_B/total_items*100:.2f}%) -> 贡献了 20.00% 的交互")
    print("="*50)

    # ==========================================
    # 绘制可用于论文的长尾分布图 (Matplotlib)
    # ==========================================
    print("3. 正在生成长尾分布图...")
    
    # 为了让图表更好看，我们对 X 轴进行归一化 (0 到 100%)
    x_percent = np.linspace(0, 100, total_items)
    
    # 配置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制主曲线：商品排名 vs 交互频次
    # 使用 log 刻度可以让长尾看得更清晰
    ax1.plot(x_percent, counts_array, color='#1f77b4', linewidth=2.5, label='Interaction Frequency')
    ax1.fill_between(x_percent[:head_item_cutoff], counts_array[:head_item_cutoff], color='#ff7f0e', alpha=0.5, label='Head (Top 20% Items)')
    ax1.fill_between(x_percent[head_item_cutoff:], counts_array[head_item_cutoff:], color='#1f77b4', alpha=0.3, label='Tail (Bottom 80% Items)')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Item Rank (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Interaction Frequency (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_title('Long-Tail Item Distribution in Amazon Toys', fontsize=16, fontweight='bold', pad=15)
    
    # 添加网格和图例
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='upper right', fontsize=12)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=600, bbox_inches='tight')
    print(f"[*] 完美！长尾分布图已保存至: {output_img_path}")

if __name__ == "__main__":
    # 确保路径指向你刚刚生成的严格对齐版 json
    SEQS_PATH = "/workspace/user_code/baseline/RQ-VAE/data/Toys/all_item_seqs.json"
    analyze_and_plot_long_tail(SEQS_PATH)