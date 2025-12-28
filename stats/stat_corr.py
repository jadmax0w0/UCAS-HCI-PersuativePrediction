import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys; sys.path.append(".")
import feat_extractors.cialdini_extractor as Cial
from feat_extractors.dim10_extractor import Dim10FeatureExtractor, DIMS_DEFAULT
from pipeline import load_cmv_data, parse_cmv_data

def analyze_all(data, feature_names=None):
    """
    主函数：执行基础统计和相关性分析
    """
    data = np.array(data)
    N, D = data.shape
    if feature_names is None:
        feature_names = [f'Feat_{i}' for i in range(D)]

    # 1. 基础单变量统计
    counts = np.sum(data, axis=0)
    means = np.mean(data, axis=0)
    df_basic = pd.DataFrame({
        'Feature': feature_names,
        'Count': counts,
        'Ratio': means,
        'Variance': means * (1 - means)
    }).set_index('Feature').sort_values('Ratio', ascending=False)

    # 2. 全局相关性矩阵 (Pearson/Phi Coefficient)
    # 对于 0/1 数据，Pearson 相关系数即为 Phi 系数
    df_corr = pd.DataFrame(data, columns=feature_names).corr()

    return df_basic, df_corr

def analyze_joint_subset(data, feature_indices, feature_names=None):
    """
    针对【特定子集】的联合统计
    例如：想看 feature[0] 和 feature[2] 同时出现的概率
    
    参数:
    feature_indices: list of int, 例如 [0, 2]
    """
    data = np.array(data)
    if feature_names is None:
        feature_names = [f'Feat_{i}' for i in range(data.shape[1])]
    
    # 提取子集名称和数据
    selected_names = [feature_names[i] for i in feature_indices]
    subset = data[:, feature_indices]
    
    # 1. AND 逻辑 (Intersection): 所有选定特征同时为 1
    # axis=1 表示沿着特征方向判断，看每一行是否全为 1
    all_true_mask = np.all(subset == 1, axis=1)
    count_and = np.sum(all_true_mask)
    ratio_and = np.mean(all_true_mask)
    
    # 2. OR 逻辑 (Union): 至少有一个特征为 1
    any_true_mask = np.any(subset == 1, axis=1)
    count_or = np.sum(any_true_mask)
    ratio_or = np.mean(any_true_mask)
    
    # 3. Jaccard 相似度 (交集 / 并集)
    # 衡量这几个特征的重合程度
    jaccard = count_and / count_or if count_or > 0 else 0.0

    # 结果整理
    result = pd.Series({
        "Selected Features": " & ".join(selected_names),
        "Sample Size (N)": data.shape[0],
        "Count (AND - All 1)": count_and,
        "Ratio (AND)": ratio_and,       # 联合概率 P(A, B, ...)
        "Count (OR - Any 1)": count_or,
        "Ratio (OR)": ratio_or,
        "Jaccard Similarity": jaccard   # 1 表示完全重合，0 表示完全不重合
    })
    
    return result


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title="Feature Correlation Matrix", mask_upper=False):
    """
    绘制相关性矩阵的热力图
    
    参数:
    corr_matrix: pd.DataFrame, 相关性矩阵 (通常是 df.corr() 的结果)
    title: str, 图表标题
    mask_upper: bool, 是否遮挡上半三角 (因为矩阵是对称的，遮挡后更简洁)
    """
    temp_corr = corr_matrix.copy()

    np.fill_diagonal(temp_corr.values, np.nan)
    
    vmin = np.nanmin(temp_corr.values)
    vmax = np.nanmax(temp_corr.values)

    diag_mask = np.eye(len(corr_matrix), dtype=bool)

    # 设置画布大小
    plt.figure(figsize=(10, 8))
    
    # 设置遮挡掩码 (可选)
    mask = diag_mask
    if mask_upper:
        mask = np.triu(np.ones_like(temp_corr, dtype=bool)) | mask

    # 绘制热力图
    sns.heatmap(
        temp_corr,
        annot=True,         # 在格子里显示数值
        fmt=".2f",          # 数值保留两位小数
        cmap="coolwarm",    # 颜色映射：冷暖色 (蓝色负相关，红色正相关)
        vmin=vmin, vmax=vmax,
        center=0,           # 0 值对应白色/灰色
        square=True,        # 强制每个格为正方形
        linewidths=0.5,     # 格子之间的分割线宽度
        cbar_kws={"shrink": 0.8}, # 颜色条稍微缩小一点
        mask=mask           # 应用遮挡 (如果有)
    )

    plt.title(title, fontsize=15)
    plt.xticks(rotation=45, ha='right') # X轴标签旋转，防止重叠
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():
    # Cialdini
    # feat_names = ["Reciprocity", "Consistency", "Social_Proof", "Authority", "Scarcity", "Liking"]
    # p_feat, n_feat = Cial.load_from_anno_file("dataset/cialdini_scores_train.json", partition='train')
    # print(f"{p_feat.shape=}, {n_feat.shape=}")

    # 10 Dim
    feat_names = DIMS_DEFAULT
    texts = load_cmv_data("dataset/train.jsonl")
    df = parse_cmv_data(texts, max_data_count=500)
    df_p, df_n = df[df['label'] == 1], df[df['label'] == 0]

    dim10 = Dim10FeatureExtractor()
    dim10.train()
    p_feat, n_feat = dim10.extract(df_p['p'].values.tolist()), dim10.extract(df_n['p'].values.tolist())

    p_basic, p_corr = analyze_all(p_feat, feat_names)
    n_basic, n_corr = analyze_all(n_feat, feat_names)

    print("Positive (successful persuade) stats:")
    print(p_basic.round(2))
    print("Positive 10-dim correlations:")
    print(p_corr.round(2))
    plot_correlation_heatmap(p_corr, "Positive 10-dim correlations", mask_upper=True)
    print()
    print("Negative (failed persuade) stats:")
    print(n_basic.round(2))
    print("Negative 10-dim correlations:")
    print(n_corr.round(2))
    plot_correlation_heatmap(n_corr, "Negative 10-dim correlations", mask_upper=True)


if __name__ == "__main__":
    main()