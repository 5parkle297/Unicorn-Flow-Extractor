"""
特征提取器 V2 - 从单向流中提取统计特征和序列特征

支持两种输出格式:
1. 统计特征 (CSV) - 用于传统机器学习
2. 序列特征 (PKL) - 用于深度学习

兼容:
- build_unidirectional_flows.py (旧版本)
- build_unidirectional_flows_v2.py (新版本)

作者: 基于CICFlowMeter和NetMamba思路
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="从单向流中提取特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取统计特征 (只包含有效特征)
  python extract_features_v2.py --flows flows.pkl --out features.csv
  
  # 提取所有特征 (兼容CICFlowMeter格式)
  python extract_features_v2.py --flows flows.pkl --out features.csv --full-features
  
  # 提取序列特征（用于深度学习）
  python extract_features_v2.py --flows flows.pkl --out features.pkl --format sequence
  
  # 同时输出统计和序列特征
  python extract_features_v2.py --flows flows.pkl --out features --format both
        """
    )
    
    parser.add_argument("--flows", required=True, help="Input flow pickle file")
    parser.add_argument("--out", required=True, help="Output file (csv or pkl)")
    parser.add_argument("--format", choices=['stats', 'sequence', 'both'],
                        default='stats', help="Output format")
    parser.add_argument("--max-seq-len", type=int, default=100,
                        help="Maximum sequence length for sequence features")
    parser.add_argument("--include-payload", action="store_true",
                        help="Include payload features")
    parser.add_argument("--full-features", action="store_true",
                        help="Output all 91 features including zeros (for CICFlowMeter compatibility)")
    parser.add_argument("--compact", action="store_true",
                        help="Output only essential features (recommended)")
    
    return parser.parse_args()


# ==================== 工具函数 ====================

def safe_div(a, b):
    """安全除法"""
    return a / b if b != 0 else 0


def safe_std(values):
    """安全标准差"""
    return float(np.std(values)) if len(values) > 1 else 0.0


def safe_mean(values):
    """安全均值"""
    return float(np.mean(values)) if len(values) > 0 else 0.0


def calculate_iat(timestamps):
    """计算包间隔时间统计"""
    if len(timestamps) <= 1:
        return {'total': 0, 'mean': 0, 'std': 0, 'max': 0, 'min': 0}
    
    iats = np.diff(sorted(timestamps))
    return {
        'total': float(np.sum(iats)),
        'mean': float(np.mean(iats)),
        'std': float(safe_std(iats)),
        'max': float(np.max(iats)),
        'min': float(np.min(iats))
    }


def hex_to_bytes(hex_str):
    """十六进制字符串转字节列表"""
    if not hex_str:
        return []
    try:
        return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
    except:
        return []


# ==================== 精简特征列表（只有有信息量的特征）====================

COMPACT_FEATURES = [
    # 标识信息
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'direction', 'flow_id',
    # 时间
    'timestamp', 'flow_duration',
    # 包统计
    'total_packets', 'total_bytes',
    # 包大小统计
    'packet_length_max', 'packet_length_min', 'packet_length_mean', 
    'packet_length_std', 'packet_length_var',
    # 速率
    'flow_bytes_per_s', 'flow_packets_per_s',
    # IAT
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'iat_total',
    # 平均值
    'average_packet_size',
    # Payload
    'payload_bytes_total', 'payload_bytes_mean', 'payload_bytes_std',
    'payload_bytes_max', 'payload_bytes_min', 
    'packets_with_payload', 'payload_ratio',
    # Header
    'header_bytes_total', 'header_bytes_mean',
]


# ==================== 统计特征提取 ====================

def extract_compact_features(flow_key, packets):
    """
    提取精简的统计特征 (只包含有信息量的特征)
    
    返回约35维特征
    """
    src_ip, dst_ip, src_port, dst_port, protocol, direction = flow_key
    
    if len(packets) == 0:
        return None
    
    # ========== 基础信息 ==========
    features = {
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'direction': direction,
        'flow_id': f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}-{direction}",
        'timestamp': packets[0].get('ts', 0),
    }
    
    # ========== 时间特征 ==========
    timestamps = [p.get('ts', 0) for p in packets]
    duration = max(timestamps) - min(timestamps) if timestamps else 0
    features['flow_duration'] = duration
    
    # ========== 包统计 ==========
    total_packets = len(packets)
    lengths = [p.get('len', 0) for p in packets]
    total_bytes = sum(lengths)
    
    features['total_packets'] = total_packets
    features['total_bytes'] = total_bytes
    
    # ========== 包大小统计 ==========
    if lengths:
        features['packet_length_max'] = max(lengths)
        features['packet_length_min'] = min(lengths)
        features['packet_length_mean'] = safe_mean(lengths)
        features['packet_length_std'] = safe_std(lengths)
        features['packet_length_var'] = float(np.var(lengths)) if len(lengths) > 1 else 0.0
    else:
        features['packet_length_max'] = 0
        features['packet_length_min'] = 0
        features['packet_length_mean'] = 0
        features['packet_length_std'] = 0
        features['packet_length_var'] = 0
    
    # ========== 速率 ==========
    features['flow_bytes_per_s'] = safe_div(total_bytes, duration) if duration > 0 else 0
    features['flow_packets_per_s'] = safe_div(total_packets, duration) if duration > 0 else 0
    
    # ========== IAT ==========
    iat_stats = calculate_iat(timestamps)
    features['flow_iat_mean'] = iat_stats['mean']
    features['flow_iat_std'] = iat_stats['std']
    features['flow_iat_max'] = iat_stats['max']
    features['flow_iat_min'] = iat_stats['min']
    features['iat_total'] = iat_stats['total']
    
    # ========== 平均值 ==========
    features['average_packet_size'] = safe_div(total_bytes, total_packets)
    
    # ========== Payload ==========
    payload_lengths = []
    for p in packets:
        payload = p.get('payload', '')
        if payload:
            payload_lengths.append(len(payload) // 2)
        else:
            payload_lengths.append(0)
    
    features['payload_bytes_total'] = sum(payload_lengths)
    features['payload_bytes_mean'] = safe_mean(payload_lengths)
    features['payload_bytes_std'] = safe_std(payload_lengths)
    features['payload_bytes_max'] = max(payload_lengths) if payload_lengths else 0
    features['payload_bytes_min'] = min(payload_lengths) if payload_lengths else 0
    features['packets_with_payload'] = sum(1 for l in payload_lengths if l > 0)
    features['payload_ratio'] = safe_div(features['packets_with_payload'], total_packets)
    
    # ========== Header ==========
    header_lengths = []
    for p in packets:
        header = p.get('header', '')
        if header:
            header_lengths.append(len(header) // 2)
        else:
            header_lengths.append(0)
    
    features['header_bytes_total'] = sum(header_lengths)
    features['header_bytes_mean'] = safe_mean(header_lengths)
    
    return features


def extract_statistical_features(flow_key, packets):
    """
    从单条流中提取统计特征
    
    返回约70维的特征向量
    """
    src_ip, dst_ip, src_port, dst_port, protocol, direction = flow_key
    
    if len(packets) == 0:
        return None
    
    # ========== 基础信息 ==========
    features = {
        # 标识信息
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'direction': direction,
        'flow_id': f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}-{direction}",
        
        # 时间信息
        'timestamp': packets[0].get('ts', 0),
    }
    
    # ========== 时间特征 ==========
    timestamps = [p.get('ts', 0) for p in packets]
    duration = max(timestamps) - min(timestamps) if timestamps else 0
    features['flow_duration'] = duration
    
    # ========== 包统计 ==========
    total_packets = len(packets)
    features['total_packets'] = total_packets
    features['total_fwd_packets'] = total_packets  # 单向流，全部为前向
    features['total_bwd_packets'] = 0
    
    # ========== 长度统计 ==========
    lengths = [p.get('len', 0) for p in packets]
    total_bytes = sum(lengths)
    
    features['total_bytes'] = total_bytes
    features['total_length_fwd_packets'] = total_bytes
    features['total_length_bwd_packets'] = 0
    
    if lengths:
        features['packet_length_max'] = max(lengths)
        features['packet_length_min'] = min(lengths)
        features['packet_length_mean'] = safe_mean(lengths)
        features['packet_length_std'] = safe_std(lengths)
        features['packet_length_var'] = float(np.var(lengths)) if len(lengths) > 1 else 0.0
        
        # Forward方向（单向流就是全部）
        features['fwd_packet_length_max'] = max(lengths)
        features['fwd_packet_length_min'] = min(lengths)
        features['fwd_packet_length_mean'] = safe_mean(lengths)
        features['fwd_packet_length_std'] = safe_std(lengths)
    else:
        for key in ['packet_length_max', 'packet_length_min', 'packet_length_mean', 
                    'packet_length_std', 'packet_length_var', 'fwd_packet_length_max',
                    'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std']:
            features[key] = 0
    
    # Backward方向（单向流为0）
    for key in ['bwd_packet_length_max', 'bwd_packet_length_min', 
                'bwd_packet_length_mean', 'bwd_packet_length_std']:
        features[key] = 0
    
    # ========== 流量速率 ==========
    features['flow_bytes_per_s'] = safe_div(total_bytes, duration) if duration > 0 else 0
    features['flow_packets_per_s'] = safe_div(total_packets, duration) if duration > 0 else 0
    
    features['fwd_bytes_per_s'] = features['flow_bytes_per_s']
    features['bwd_bytes_per_s'] = 0
    features['fwd_packets_per_s'] = features['flow_packets_per_s']
    features['bwd_packets_per_s'] = 0
    
    # ========== IAT 特征 ==========
    iat_stats = calculate_iat(timestamps)
    
    features['flow_iat_mean'] = iat_stats['mean']
    features['flow_iat_std'] = iat_stats['std']
    features['flow_iat_max'] = iat_stats['max']
    features['flow_iat_min'] = iat_stats['min']
    
    features['fwd_iat_total'] = iat_stats['total']
    features['fwd_iat_mean'] = iat_stats['mean']
    features['fwd_iat_std'] = iat_stats['std']
    features['fwd_iat_max'] = iat_stats['max']
    features['fwd_iat_min'] = iat_stats['min']
    
    # Backward IAT（单向流为0）
    for key in ['bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min']:
        features[key] = 0
    
    # ========== 平均值 ==========
    features['average_packet_size'] = safe_div(total_bytes, total_packets)
    features['fwd_segment_size_avg'] = features['average_packet_size']
    features['bwd_segment_size_avg'] = 0
    
    # ========== 比率 ==========
    features['down_up_ratio'] = 0  # 单向流无法计算
    
    # ========== Payload 统计 ==========
    payload_lengths = []
    for p in packets:
        payload = p.get('payload', '')
        if payload:
            payload_lengths.append(len(payload) // 2)  # hex字符数 / 2 = 字节数
        else:
            payload_lengths.append(0)
    
    features['payload_bytes_total'] = sum(payload_lengths)
    features['payload_bytes_mean'] = safe_mean(payload_lengths)
    features['payload_bytes_std'] = safe_std(payload_lengths)
    features['payload_bytes_max'] = max(payload_lengths) if payload_lengths else 0
    features['payload_bytes_min'] = min(payload_lengths) if payload_lengths else 0
    features['packets_with_payload'] = sum(1 for l in payload_lengths if l > 0)
    features['payload_ratio'] = safe_div(features['packets_with_payload'], total_packets)
    
    # ========== Header 统计（如果有）==========
    header_lengths = []
    for p in packets:
        header = p.get('header', '')
        if header:
            header_lengths.append(len(header) // 2)
        else:
            header_lengths.append(0)
    
    features['header_bytes_total'] = sum(header_lengths)
    features['header_bytes_mean'] = safe_mean(header_lengths)
    
    # ========== 子流特征（简化）==========
    features['subflow_fwd_packets'] = total_packets
    features['subflow_fwd_bytes'] = total_bytes
    features['subflow_bwd_packets'] = 0
    features['subflow_bwd_bytes'] = 0
    
    # ========== 占位特征（保持与CICFlowMeter兼容）==========
    for key in ['fin_flag_count', 'syn_flag_count', 'rst_flag_count', 
                'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
                'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
                'fwd_header_length', 'bwd_header_length',
                'fwd_bytes_bulk_avg', 'fwd_packet_bulk_avg', 'fwd_bulk_rate_avg',
                'bwd_bytes_bulk_avg', 'bwd_packet_bulk_avg', 'bwd_bulk_rate_avg',
                'active_mean', 'active_std', 'active_max', 'active_min',
                'idle_mean', 'idle_std', 'idle_max', 'idle_min']:
        features[key] = 0
    
    return features


# ==================== 序列特征提取 ====================

def extract_sequence_features(flow_key, packets, max_len=100, include_payload=False):
    """
    从单条流中提取序列特征
    
    用于深度学习模型（LSTM, Transformer等）
    
    返回:
        - packet_lengths: 包大小序列
        - iats: 包间隔时间序列
        - payload_bytes: (可选) payload字节序列
    """
    src_ip, dst_ip, src_port, dst_port, protocol, direction = flow_key
    
    if len(packets) == 0:
        return None
    
    # 限制长度
    packets = packets[:max_len]
    n_packets = len(packets)
    
    # ========== 基础序列 ==========
    # 包大小序列
    lengths = [p.get('len', 0) for p in packets]
    
    # 时间戳序列
    timestamps = [p.get('ts', 0) for p in packets]
    
    # IAT序列
    if len(timestamps) > 1:
        iats = [0.0] + list(np.diff(timestamps))
    else:
        iats = [0.0]
    
    # 方向序列
    directions = [p.get('direction', 0) for p in packets]
    
    # ========== 可选: Payload字节序列 ==========
    payload_bytes = None
    header_bytes = None
    
    if include_payload:
        # 提取payload字节
        payload_list = []
        for p in packets:
            payload = p.get('payload', '')
            if payload:
                bytes_list = hex_to_bytes(payload)
                payload_list.append(bytes_list)
            else:
                payload_list.append([])
        payload_bytes = payload_list
        
        # 提取header字节
        header_list = []
        for p in packets:
            header = p.get('header', '')
            if header:
                bytes_list = hex_to_bytes(header)
                header_list.append(bytes_list)
            else:
                header_list.append([])
        header_bytes = header_list
    
    # ========== 构建特征字典 ==========
    features = {
        'flow_key': flow_key,
        'flow_id': f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}-{direction}",
        'n_packets': n_packets,
        
        # 序列特征
        'packet_lengths': lengths,
        'iats': iats,
        'directions': directions,
        'timestamps': timestamps,
        
        # 流级别元信息
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'direction': direction,
    }
    
    # 可选特征
    if payload_bytes is not None:
        features['payload_bytes'] = payload_bytes
    if header_bytes is not None:
        features['header_bytes'] = header_bytes
    
    return features


# ==================== 数据加载 ====================

def load_flows(filepath):
    """
    加载流数据，兼容新旧格式
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 检查是否为新格式
    if isinstance(data, dict) and 'flows' in data:
        # 新格式
        flows = data['flows']
        config = data.get('config', {})
        stats = data.get('stats', {})
        netmamba_flows = data.get('netmamba_flows', None)
        
        print(f"[+] 检测到新版本格式")
        print(f"    配置: {config}")
        print(f"    统计: {stats}")
        
        return flows, config, netmamba_flows
    else:
        # 旧格式：直接就是flows字典
        print(f"[+] 检测到旧版本格式")
        return data, {}, None


# ==================== 主程序 ====================

def main():
    args = parse_args()
    
    print("=" * 60)
    print("特征提取器 V2")
    print("=" * 60)
    
    # 加载数据
    print(f"\n[+] 加载流数据: {args.flows}")
    flows, config, netmamba_flows = load_flows(args.flows)
    print(f"[+] 总流数: {len(flows):,}")
    
    # ========== 统计特征 ==========
    if args.format in ['stats', 'both']:
        # 选择提取函数
        use_compact = args.compact or (not args.full_features)  # 默认使用精简版
        
        if use_compact:
            print(f"\n[+] 提取精简统计特征 (只包含有效特征)...")
            extract_func = extract_compact_features
        else:
            print(f"\n[+] 提取完整统计特征 (CICFlowMeter兼容格式)...")
            extract_func = extract_statistical_features
        
        stats_features = []
        for i, (key, packets) in enumerate(flows.items(), 1):
            if i % 5000 == 0:
                print(f"    处理进度: {i:,}/{len(flows):,}")
            
            features = extract_func(key, packets)
            if features:
                stats_features.append(features)
        
        print(f"[+] 成功提取: {len(stats_features):,} 条")
        
        # 转换为DataFrame
        df = pd.DataFrame(stats_features)
        
        # 输出
        if args.format == 'stats':
            output_path = args.out if args.out.endswith('.csv') else args.out + '.csv'
        else:
            output_path = args.out + '_stats.csv'
        
        print(f"[+] 保存统计特征到: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"    形状: {df.shape}")
        print(f"    特征列: {len(df.columns)}")
        
        # 分析特征
        numeric_cols = df.select_dtypes(include=['number']).columns
        zero_cols = [col for col in numeric_cols if (df[col] == 0).all()]
        useful_cols = [col for col in numeric_cols if not (df[col] == 0).all()]
        
        print(f"\n[+] 特征分析:")
        print(f"    总数值特征: {len(numeric_cols)}")
        print(f"    有效特征: {len(useful_cols)}")
        print(f"    全0特征: {len(zero_cols)}")
        
        if zero_cols and len(zero_cols) <= 10:
            print(f"\n[!] 全0特征列表: {zero_cols}")
    
    # ========== 序列特征 ==========
    if args.format in ['sequence', 'both']:
        print(f"\n[+] 提取序列特征...")
        print(f"    最大序列长度: {args.max_seq_len}")
        print(f"    包含Payload: {args.include_payload}")
        
        seq_features = []
        for i, (key, packets) in enumerate(flows.items(), 1):
            if i % 5000 == 0:
                print(f"    处理进度: {i:,}/{len(flows):,}")
            
            features = extract_sequence_features(
                key, packets, 
                max_len=args.max_seq_len,
                include_payload=args.include_payload
            )
            if features:
                seq_features.append(features)
        
        print(f"[+] 成功提取: {len(seq_features):,} 条")
        
        # 输出
        if args.format == 'sequence':
            output_path = args.out if args.out.endswith('.pkl') else args.out + '.pkl'
        else:
            output_path = args.out + '_sequence.pkl'
        
        print(f"[+] 保存序列特征到: {output_path}")
        
        output_data = {
            'sequences': seq_features,
            'config': {
                'max_seq_len': args.max_seq_len,
                'include_payload': args.include_payload,
                'n_flows': len(seq_features)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        # 显示样例
        if seq_features:
            sample = seq_features[0]
            print(f"\n[+] 序列特征样例:")
            print(f"    Flow ID: {sample['flow_id']}")
            print(f"    包数: {sample['n_packets']}")
            print(f"    包大小序列: {sample['packet_lengths'][:5]}...")
            print(f"    IAT序列: {[round(x, 6) for x in sample['iats'][:5]]}...")
    
    # ========== 显示NetMamba格式（如果有）==========
    if netmamba_flows:
        print(f"\n[+] NetMamba格式数据:")
        print(f"    流数: {len(netmamba_flows):,}")
        sample_key = list(netmamba_flows.keys())[0]
        sample_array = netmamba_flows[sample_key]
        print(f"    数组形状: {sample_array.shape}")
        print(f"    数据类型: {sample_array.dtype}")
    
    print("\n[+] 完成!")


if __name__ == "__main__":
    main()
