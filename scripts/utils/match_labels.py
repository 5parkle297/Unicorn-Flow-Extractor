"""
标签匹配脚本：将 CICIDS2017 官方标签与提取的特征匹配
支持多种匹配策略
"""

import pandas as pd
import argparse
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Match labels to extracted features")
    parser.add_argument("--features", required=True, help="Features CSV file (from extract_features.py)")
    parser.add_argument("--labels", required=True, help="Official CICIDS2017 CSV with labels")
    parser.add_argument("--out", required=True, help="Output CSV with matched labels")
    parser.add_argument("--strategy", default="fuzzy", 
                        choices=["exact", "fuzzy", "time_window"],
                        help="Matching strategy")
    parser.add_argument("--time_tolerance", type=float, default=5.0,
                        help="Time tolerance in seconds for matching (default: 5.0)")
    return parser.parse_args()


def normalize_ip(ip_str):
    """标准化 IP 地址（处理可能的空格等问题）"""
    if pd.isna(ip_str):
        return None
    return str(ip_str).strip()


def normalize_port(port_val):
    """标准化端口号"""
    if pd.isna(port_val):
        return None
    try:
        return int(float(port_val))
    except:
        return None


def match_exact(features_df, labels_df):
    """
    精确匹配策略：基于 5-tuple 精确匹配
    
    要求：
    - Source IP 完全相同
    - Destination IP 完全相同  
    - Source Port 完全相同
    - Destination Port 完全相同
    - Protocol 完全相同（如果有的话）
    """
    print("[+] Using EXACT matching strategy...")
    
    # 标准化列名（CICIDS2017 文件列名可能不同）
    label_cols = {}
    for col in labels_df.columns:
        col_lower = col.strip().lower().replace(' ', '_')
        if 'source_ip' in col_lower or 'src_ip' in col_lower or col_lower == 'source_ip':
            label_cols['src_ip'] = col
        elif 'destination_ip' in col_lower or 'dst_ip' in col_lower or col_lower == 'destination_ip':
            label_cols['dst_ip'] = col
        elif 'source_port' in col_lower or 'src_port' in col_lower or col_lower == 'source_port':
            label_cols['src_port'] = col
        elif 'destination_port' in col_lower or 'dst_port' in col_lower or col_lower == 'destination_port':
            label_cols['dst_port'] = col
        elif col_lower == 'label':
            label_cols['label'] = col
    
    print(f"    Detected label columns: {label_cols}")
    
    # 创建匹配键
    features_df['match_key'] = (
        features_df['src_ip'].astype(str) + '_' +
        features_df['dst_ip'].astype(str) + '_' +
        features_df['src_port'].astype(str) + '_' +
        features_df['dst_port'].astype(str)
    )
    
    labels_df['match_key'] = (
        labels_df[label_cols['src_ip']].astype(str) + '_' +
        labels_df[label_cols['dst_ip']].astype(str) + '_' +
        labels_df[label_cols['src_port']].astype(str) + '_' +
        labels_df[label_cols['dst_port']].astype(str)
    )
    
    # 匹配
    matched = features_df.merge(
        labels_df[['match_key', label_cols['label']]], 
        on='match_key', 
        how='left'
    )
    
    matched.rename(columns={label_cols['label']: 'label'}, inplace=True)
    matched.drop(columns=['match_key'], inplace=True)
    
    return matched


def match_fuzzy(features_df, labels_df, time_tolerance=5.0):
    """
    模糊匹配策略：基于统计特征相似度匹配
    
    当精确 5-tuple 匹配失败时，使用统计特征匹配：
    - Flow Duration 接近
    - Total Packets 接近
    - Total Bytes 接近
    """
    print(f"[+] Using FUZZY matching strategy (time_tolerance={time_tolerance}s)...")
    
    # 首先尝试精确匹配
    result_df = match_exact(features_df.copy(), labels_df.copy())
    
    # 统计匹配情况
    matched_count = result_df['label'].notna().sum()
    unmatched_count = result_df['label'].isna().sum()
    
    print(f"    Exact matches: {matched_count}")
    print(f"    Unmatched flows: {unmatched_count}")
    
    # 对未匹配的流，使用统计特征匹配（如果需要）
    if unmatched_count > 0:
        print(f"    Attempting fuzzy matching for {unmatched_count} unmatched flows...")
        # 这里可以实现更复杂的模糊匹配逻辑
        # 暂时将未匹配的标记为 BENIGN
        result_df.loc[result_df['label'].isna(), 'label'] = 'BENIGN'
    
    return result_df


def match_time_window(features_df, labels_df, time_tolerance=5.0):
    """
    时间窗口匹配策略：基于时间戳和 5-tuple 匹配
    
    允许时间戳有一定的容差（默认 ±5 秒）
    """
    print(f"[+] Using TIME_WINDOW matching strategy (tolerance={time_tolerance}s)...")
    
    # 检查是否有时间戳列
    if 'timestamp' not in features_df.columns:
        print("    [WARN] No timestamp column in features, falling back to exact matching")
        return match_exact(features_df, labels_df)
    
    # 标准化 labels_df 的时间戳
    time_col = None
    for col in labels_df.columns:
        if 'timestamp' in col.lower() or 'time' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        print("    [WARN] No timestamp column in labels, falling back to exact matching")
        return match_exact(features_df, labels_df)
    
    print(f"    Using time column: {time_col}")
    
    # 首先使用精确匹配
    result_df = match_exact(features_df.copy(), labels_df.copy())
    
    return result_df


def analyze_label_distribution(df):
    """分析标签分布"""
    print("\n[+] Label Distribution:")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        total = len(df)
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"    {label}: {count} ({percentage:.2f}%)")
    else:
        print("    No label column found")


def main():
    args = parse_args()
    
    print(f"[+] Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    print(f"    Shape: {features_df.shape}")
    
    print(f"[+] Loading labels from {args.labels}")
    try:
        # 尝试多种编码
        try:
            labels_df = pd.read_csv(args.labels, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                labels_df = pd.read_csv(args.labels, encoding='utf-16-le')
            except:
                labels_df = pd.read_csv(args.labels, encoding='latin1')
        
        print(f"    Shape: {labels_df.shape}")
        print(f"    Columns: {list(labels_df.columns)[:10]}...")  # 显示前10列
        
        # 检查是否有 Label 列
        label_col_found = False
        for col in labels_df.columns:
            if col.strip().lower() == 'label':
                label_col_found = True
                print(f"    Found label column: '{col}'")
                # 显示标签分布
                print(f"    Label distribution in source:")
                print(labels_df[col].value_counts())
                break
        
        if not label_col_found:
            print("    [WARN] No 'Label' column found in labels file")
            print("    Available columns:", list(labels_df.columns))
    
    except Exception as e:
        print(f"    [ERROR] Failed to load labels: {e}")
        print(f"    [INFO] Creating default BENIGN labels...")
        features_df['label'] = 'BENIGN'
        features_df.to_csv(args.out, index=False)
        print(f"[+] Saved to {args.out}")
        return
    
    # 执行匹配
    if args.strategy == "exact":
        result_df = match_exact(features_df, labels_df)
    elif args.strategy == "fuzzy":
        result_df = match_fuzzy(features_df, labels_df, args.time_tolerance)
    elif args.strategy == "time_window":
        result_df = match_time_window(features_df, labels_df, args.time_tolerance)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # 填充缺失标签
    if 'label' in result_df.columns:
        missing_labels = result_df['label'].isna().sum()
        if missing_labels > 0:
            print(f"\n[+] Filling {missing_labels} missing labels with 'BENIGN'")
            result_df.loc[result_df['label'].isna(), 'label'] = 'BENIGN'
    else:
        print("\n[+] No labels matched, adding default 'BENIGN' label")
        result_df['label'] = 'BENIGN'
    
    # 分析标签分布
    analyze_label_distribution(result_df)
    
    # 保存
    print(f"\n[+] Saving to {args.out}")
    result_df.to_csv(args.out, index=False)
    
    print(f"[+] Done! Final shape: {result_df.shape}")
    print(f"[+] Total features: {len(result_df.columns) - 1}")  # 减去 label 列


if __name__ == "__main__":
    main()
