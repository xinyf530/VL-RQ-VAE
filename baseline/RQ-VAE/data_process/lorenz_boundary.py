import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_lorenz_and_gini(seqs_path, output_img_path="lorenz_curve_boundaries.png"):
    print(f"1. 正在加载交互数据: {seqs_path} ...")
    with open(seqs_path, 'r', encoding='utf-8') as f:
        all_item_seqs = json.load(f)
        
    # 统计频次
    item_counts = Counter()
    for user, seq in all_item_seqs.items():
        item_counts.update(seq)
        
    # 洛伦兹曲线需要从小到大排序 (Ascending Order)
    counts_ascending = np.array(sorted(item_counts.values()))
    total_items = len(counts_ascending)
    total_interactions = np.sum(counts_ascending)
    global_average = total_interactions / total_items
    
    # ---------------------------------------------------------
    # 核心数学计算：洛伦兹曲线坐标
    # ---------------------------------------------------------
    # X轴：累计物品比例 (0 到 1)
    cum_items_prop = np.arange(1, total_items + 1) / total_items
    # Y轴：累计交互比例 (0 到 1)
    cum_interactions_prop = np.cumsum(counts_ascending) / total_interactions
    
    # 补充原点 (0,0)
    cum_items_prop = np.insert(cum_items_prop, 0, 0.0)
    cum_interactions_prop = np.insert(cum_interactions_prop, 0, 0.0)
    
    # ---------------------------------------------------------
    # 核心数学计算：基尼系数 (Gini Coefficient)
    # Gini = 1 - 2 * 洛伦兹曲线下方面积
    # ---------------------------------------------------------
    area_under_lorenz = np.trapz(cum_interactions_prop, cum_items_prop)
    gini_coefficient = 1.0 - 2.0 * area_under_lorenz
    
    # ---------------------------------------------------------
    # 数学边界切分：Head / Mid / Tail
    # ---------------------------------------------------------
    # 【Mid/Tail 边界】：洛伦兹曲线上斜率等于 1 的点 (即频次刚好等于平均值的点)
    # 因为 counts 是从小到大排的，找到第一个 >= 平均值的物品索引
    tail_cutoff_idx = np.argmax(counts_ascending >= global_average)
    tail_items_count = tail_cutoff_idx
    tail_items_prop = tail_items_count / total_items
    tail_interactions_prop = cum_interactions_prop[tail_cutoff_idx]
    
    # 【Head/Mid 边界】：帕累托法则，占据最顶端 80% 交互的物品作为 Head
    # 在从小到大排序的累积数组中，找累积交互比例刚好达到 20% 的点 (因为顶端占了80%)
    head_cutoff_idx = np.argmax(cum_interactions_prop >= 0.20)
    head_items_count = total_items - head_cutoff_idx
    head_items_prop = head_items_count / total_items
    
    mid_items_count = total_items - head_items_count - tail_items_count
    mid_items_prop = mid_items_count / total_items
    
    # ---------------------------------------------------------
    # 终端打印学术报告
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("📈 基于数学定义的 Head/Mid/Tail 严谨划分报告")
    print("="*60)
    print(f"全局平均交互次数: {global_average:.2f} 次/物品")
    print(f"基尼系数 (Gini):  {gini_coefficient:.4f} (越接近1越极端)")
    if gini_coefficient > 0.8:
        print("结论: 基尼系数 > 0.8，存在极端的长尾不平等现象，急需专门的优化策略！")
    print("-" * 60)
    print(f"【Tail (尾部)】定义: 交互量低于全局平均值的底层商品")
    print(f"  -> 涵盖前 {tail_items_prop*100:.2f}% 的物品 (共 {tail_items_count:,} 个)")
    print(f"  -> 仅占总交互量的 {tail_interactions_prop*100:.2f}%")
    print(f"【Mid (腰部)】定义: 高于平均值，但未进入帕累托头部的中坚商品")
    print(f"  -> 涵盖中间 {mid_items_prop*100:.2f}% 的物品 (共 {mid_items_count:,} 个)")
    print(f"【Head (头部)】定义: 占据系统总交互量 80% 的顶端商品 (Pareto Principle)")
    print(f"  -> 涵盖最后 {head_items_prop*100:.2f}% 的物品 (共 {head_items_count:,} 个)")
    print("="*60)

    # ---------------------------------------------------------
    # 绘制高级学术图表：洛伦兹曲线
    # ---------------------------------------------------------
    print("3. 正在生成高级洛伦兹曲线边界图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绝对公平线
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=2, label='Line of Perfect Equality')
    
    # 洛伦兹曲线
    ax.plot(cum_items_prop, cum_interactions_prop, color='#2ca02c', linewidth=3, label='Lorenz Curve')
    
    # 填充颜色 (Tail / Mid / Head)
    ax.fill_between(cum_items_prop[:tail_cutoff_idx+1], cum_interactions_prop[:tail_cutoff_idx+1], 
                    color='#1f77b4', alpha=0.3, label=f'Tail (Below Average)')
    ax.fill_between(cum_items_prop[tail_cutoff_idx:head_cutoff_idx+1], cum_interactions_prop[tail_cutoff_idx:head_cutoff_idx+1], 
                    color='#ff7f0e', alpha=0.3, label='Mid (Torso)')
    ax.fill_between(cum_items_prop[head_cutoff_idx:], cum_interactions_prop[head_cutoff_idx:], 
                    color='#d62728', alpha=0.3, label='Head (Top 80% Interactions)')
    
    # 标记切点
    ax.scatter([cum_items_prop[tail_cutoff_idx]], [cum_interactions_prop[tail_cutoff_idx]], color='blue', s=100, zorder=5)
    ax.annotate(f'Tangent Slope=1\n(Avg: {global_average:.1f})', 
                xy=(cum_items_prop[tail_cutoff_idx], cum_interactions_prop[tail_cutoff_idx]), 
                xytext=(-60, 40), textcoords='offset points', arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=11, fontweight='bold')

    ax.set_title(f'Lorenz Curve & Mathematical Boundaries (Gini = {gini_coefficient:.3f})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cumulative Share of Items (From Least to Most Popular)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Share of Interactions', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"[*] 洛伦兹曲线已生成: {output_img_path}")

if __name__ == "__main__":
    SEQS_PATH = "../data/Toys/all_item_seqs.json"
    analyze_lorenz_and_gini(SEQS_PATH)