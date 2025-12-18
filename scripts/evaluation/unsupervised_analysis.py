"""
无监督学习分析脚本 - 用于没有标签的CICIDS2017数据

支持的方法:
1. 聚类分析 (K-Means, DBSCAN)
2. 异常检测 (Isolation Forest, Local Outlier Factor)
3. 降维可视化 (PCA, t-SNE, UMAP)
4. 自动编码器 (用于特征学习)

注意: CICIDS2017的Monday数据是纯正常流量（没有攻击）
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="无监督学习分析")
    parser.add_argument("--input", required=True, help="输入文件 (pkl或csv)")
    parser.add_argument("--method", default="all", 
                       choices=["cluster", "anomaly", "visualize", "all"],
                       help="分析方法")
    parser.add_argument("--n-clusters", type=int, default=5, help="聚类数量")
    parser.add_argument("--contamination", type=float, default=0.01, 
                       help="异常比例 (用于异常检测)")
    parser.add_argument("--output-dir", default="analysis_results", help="输出目录")
    return parser.parse_args()


def load_data(filepath):
    """加载数据"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 检查是否是新格式
        if isinstance(data, dict) and 'flows' in data:
            flows = data['flows']
            netmamba_flows = data.get('netmamba_flows', None)
            
            print(f"[+] 从PKL加载数据")
            print(f"    流数量: {len(flows):,}")
            
            # 提取基本特征用于分析
            features = []
            for key, packets in flows.items():
                if len(packets) > 0:
                    lengths = [p.get('len', 0) for p in packets]
                    timestamps = [p.get('ts', 0) for p in packets]
                    
                    # 计算IAT
                    if len(timestamps) > 1:
                        iats = np.diff(sorted(timestamps))
                        iat_mean = np.mean(iats)
                        iat_std = np.std(iats)
                    else:
                        iat_mean = iat_std = 0
                    
                    feat = {
                        'n_packets': len(packets),
                        'total_bytes': sum(lengths),
                        'pkt_len_mean': np.mean(lengths),
                        'pkt_len_std': np.std(lengths) if len(lengths) > 1 else 0,
                        'pkt_len_max': max(lengths),
                        'pkt_len_min': min(lengths),
                        'iat_mean': iat_mean,
                        'iat_std': iat_std,
                        'duration': max(timestamps) - min(timestamps) if timestamps else 0,
                        'protocol': key[4],  # protocol from flow key
                        'src_port': key[2],
                        'dst_port': key[3],
                    }
                    features.append(feat)
            
            df = pd.DataFrame(features)
            return df, netmamba_flows
        else:
            # 旧格式
            flows = data
            print(f"[+] 从PKL加载旧格式数据: {len(flows):,} 流")
            return None, None
    
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        print(f"[+] 从CSV加载数据: {df.shape}")
        return df, None
    
    else:
        raise ValueError(f"不支持的文件格式: {filepath.suffix}")


def prepare_features(df, feature_cols=None):
    """准备特征矩阵"""
    if feature_cols is None:
        # 选择数值列，排除标识列
        exclude_cols = ['src_ip', 'dst_ip', 'flow_id', 'timestamp', 'label']
        feature_cols = [c for c in df.columns 
                       if df[c].dtype in ['int64', 'float64'] 
                       and c not in exclude_cols]
    
    X = df[feature_cols].values
    
    # 处理无穷值和NaN
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols


def clustering_analysis(X, n_clusters=5):
    """聚类分析"""
    print("\n" + "=" * 50)
    print("聚类分析")
    print("=" * 50)
    
    results = {}
    
    # K-Means
    print("\n[1] K-Means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    results['kmeans'] = kmeans_labels
    
    # 统计每个簇的大小
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    print(f"    簇分布: {dict(zip(unique, counts))}")
    
    # DBSCAN
    print("\n[2] DBSCAN聚类...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    results['dbscan'] = dbscan_labels
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"    发现簇数: {n_clusters_dbscan}")
    print(f"    噪声点: {n_noise} ({n_noise/len(X)*100:.2f}%)")
    
    return results


def anomaly_detection(X, contamination=0.01):
    """异常检测"""
    print("\n" + "=" * 50)
    print("异常检测")
    print("=" * 50)
    
    results = {}
    
    # Isolation Forest
    print(f"\n[1] Isolation Forest (contamination={contamination})...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso_forest.fit_predict(X)
    iso_scores = iso_forest.score_samples(X)
    
    n_anomalies = sum(iso_labels == -1)
    print(f"    检测到异常: {n_anomalies} ({n_anomalies/len(X)*100:.2f}%)")
    results['isolation_forest'] = {'labels': iso_labels, 'scores': iso_scores}
    
    # Local Outlier Factor
    print(f"\n[2] Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_
    
    n_anomalies_lof = sum(lof_labels == -1)
    print(f"    检测到异常: {n_anomalies_lof} ({n_anomalies_lof/len(X)*100:.2f}%)")
    results['lof'] = {'labels': lof_labels, 'scores': lof_scores}
    
    return results


def visualization(X, cluster_labels=None, output_dir='analysis_results'):
    """降维可视化"""
    print("\n" + "=" * 50)
    print("降维可视化")
    print("=" * 50)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # PCA
    print("\n[1] PCA降维...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"    解释方差比: {pca.explained_variance_ratio_}")
    
    # t-SNE (采样以加速)
    print("\n[2] t-SNE降维...")
    if len(X) > 5000:
        # 采样
        indices = np.random.choice(len(X), 5000, replace=False)
        X_sample = X[indices]
        labels_sample = cluster_labels[indices] if cluster_labels is not None else None
    else:
        X_sample = X
        labels_sample = cluster_labels
        indices = np.arange(len(X))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA图
    if cluster_labels is not None:
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                  cmap='tab10', alpha=0.5, s=5)
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
    else:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=5)
    axes[0].set_title('PCA Visualization')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # t-SNE图
    if labels_sample is not None:
        scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, 
                                  cmap='tab10', alpha=0.5, s=5)
        plt.colorbar(scatter, ax=axes[1], label='Cluster')
    else:
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5, s=5)
    axes[1].set_title('t-SNE Visualization')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'visualization.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    已保存: {output_path}")
    
    return X_pca, X_tsne


def feature_importance_analysis(df, cluster_labels, feature_cols):
    """特征重要性分析（基于聚类结果）"""
    print("\n" + "=" * 50)
    print("特征重要性分析")
    print("=" * 50)
    
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # 计算每个特征在不同簇之间的差异
    print("\n各簇特征均值:")
    cluster_means = df_analysis.groupby('cluster')[feature_cols].mean()
    print(cluster_means.round(2))
    
    # 计算特征的方差贡献
    print("\n特征的簇间方差:")
    variances = cluster_means.var()
    variances_sorted = variances.sort_values(ascending=False)
    for feat, var in variances_sorted.head(10).items():
        print(f"    {feat}: {var:.4f}")


def save_results(df, cluster_labels, anomaly_results, output_dir):
    """保存分析结果"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 添加聚类标签
    df_result = df.copy()
    df_result['cluster_kmeans'] = cluster_labels.get('kmeans')
    df_result['cluster_dbscan'] = cluster_labels.get('dbscan')
    
    # 添加异常检测结果
    if 'isolation_forest' in anomaly_results:
        df_result['anomaly_iforest'] = anomaly_results['isolation_forest']['labels']
        df_result['anomaly_score_iforest'] = anomaly_results['isolation_forest']['scores']
    
    if 'lof' in anomaly_results:
        df_result['anomaly_lof'] = anomaly_results['lof']['labels']
        df_result['anomaly_score_lof'] = anomaly_results['lof']['scores']
    
    # 保存
    output_path = Path(output_dir) / 'analysis_results.csv'
    df_result.to_csv(output_path, index=False)
    print(f"\n[+] 结果已保存到: {output_path}")
    
    return df_result


def main():
    args = parse_args()
    
    print("=" * 60)
    print("无监督学习分析")
    print("=" * 60)
    
    # 加载数据
    print(f"\n[+] 加载数据: {args.input}")
    df, netmamba_flows = load_data(args.input)
    
    if df is None:
        print("[!] 无法加载数据")
        return
    
    print(f"[+] 数据形状: {df.shape}")
    print(f"[+] 特征列: {list(df.columns)}")
    
    # 准备特征
    X, feature_cols = prepare_features(df)
    print(f"[+] 特征矩阵: {X.shape}")
    
    cluster_results = {}
    anomaly_results = {}
    
    # 聚类分析
    if args.method in ['cluster', 'all']:
        cluster_results = clustering_analysis(X, n_clusters=args.n_clusters)
    
    # 异常检测
    if args.method in ['anomaly', 'all']:
        anomaly_results = anomaly_detection(X, contamination=args.contamination)
    
    # 可视化
    if args.method in ['visualize', 'all']:
        cluster_labels = cluster_results.get('kmeans')
        visualization(X, cluster_labels, args.output_dir)
    
    # 特征重要性
    if cluster_results:
        feature_importance_analysis(df, cluster_results['kmeans'], feature_cols)
    
    # 保存结果
    if cluster_results or anomaly_results:
        save_results(df, cluster_results, anomaly_results, args.output_dir)
    
    print("\n[+] 分析完成!")


if __name__ == "__main__":
    main()
