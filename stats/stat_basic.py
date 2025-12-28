import numpy as np
import pandas as pd

import sys; sys.path.append(".")
import feat_extractors.cialdini_extractor as Cial


def analyze_binary_features(data, feature_names=None):
    """
    计算 N x D 0/1 矩阵的每一维统计特征
    
    参数:
    data: np.array, 形状为 [N, D]
    feature_names: list, 可选，特征名称列表，长度需为 D
    
    返回:
    pd.DataFrame: 包含每一维统计信息的表格
    """
    # 确保输入是 numpy 数组
    data = np.array(data)
    N, D = data.shape
    
    # 如果没有提供特征名，生成默认名称 Feature_0, Feature_1...
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(D)]
    
    # 1. 计算基础统计量
    # axis=0 表示沿着 N 的方向（列）进行计算
    counts_1 = np.sum(data, axis=0)      # 1 的个数
    ratios_1 = np.mean(data, axis=0)     # 1 的占比 (即概率 p)
    counts_0 = N - counts_1              # 0 的个数
    
    # 2. 计算方差 (对于伯努利分布，方差 = p * (1-p))
    variances = np.var(data, axis=0)
    
    # 3. 组装成 DataFrame 以便查看
    stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Count_1 (True)': counts_1,
        'Count_0 (False)': counts_0,
        'Ratio_1 (Mean)': ratios_1,       # 这一列最重要，代表特征出现的频率
        'Ratio_0': 1 - ratios_1,
        'Variance': variances
    })
    
    # 设置特征名为索引，方便查询
    stats_df.set_index('Feature', inplace=True)
    
    # 按 1 的占比降序排列，方便观察哪些特征最常见
    stats_df.sort_values(by='Ratio_1 (Mean)', ascending=False, inplace=True)
    
    return stats_df


def main():
    p_cial, n_cial = Cial.load_from_anno_file("dataset/cialdini_scores_train.json", partition='train')
    print(f"{p_cial.shape=}, {n_cial.shape=}")
    
    p_stats = analyze_binary_features(p_cial, ["Reciprocity", "Consistency", "Social_Proof", "Authority", "Scarcity", "Liking"])
    n_stats = analyze_binary_features(n_cial, ["Reciprocity", "Consistency", "Social_Proof", "Authority", "Scarcity", "Liking"])

    print("Positive (successful persuade) stats:")
    print(p_stats.round(3))
    print()
    print("Negative (failed persuade) stats:")
    print(n_stats.round(3))


if __name__ == "__main__":
    main()