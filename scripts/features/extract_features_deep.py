#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度特征提取器 - 四层特征金字塔
基于NetMamba和多个前沿研究

Layer 1: 增强统计特征 (~100维)
Layer 2: 时序行为特征  
Layer 3: 字节级序列特征 (1600字节)
Layer 4: 深度语义特征 (~100维)
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from typing import Dict, List, Any
import sys

# ============================================================
# Layer 1: 增强统计特征
# ============================================================

class EnhancedStatisticalFeatures:
    """增强统计特征提取器"""
    
    def extract(self, packets: List[Dict]) -> Dict[str, float]:
        """提取增强统计特征"""
        if not packets:
            return self._get_empty_features()
        
        features = {}
        
        # 基础统计
        lengths = [p['len'] for p in packets]
        features['total_packets'] = len(packets)
        features['total_bytes'] = sum(lengths)
        features['min_pkt_len'] = min(lengths)
        features['max_pkt_len'] = max(lengths)
        features['mean_pkt_len'] = np.mean(lengths)
        features['std_pkt_len'] = np.std(lengths)
        
        # 时间统计
        timestamps = [p['ts'] for p in packets]
        features['duration'] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        # IAT（包间隔时间）
        if len(timestamps) > 1:
            iats = np.diff(timestamps)
            features['mean_iat'] = np.mean(iats)
            features['std_iat'] = np.std(iats)
            features['min_iat'] = np.min(iats)
            features['max_iat'] = np.max(iats)
            features['iat_cv'] = features['std_iat'] / features['mean_iat'] if features['mean_iat'] > 0 else 0
        else:
            features['mean_iat'] = 0
            features['std_iat'] = 0
            features['min_iat'] = 0
            features['max_iat'] = 0
            features['iat_cv'] = 0
        
        # 高阶统计
        if len(lengths) > 2:
            features['pkt_len_skewness'] = stats.skew(lengths)
            features['pkt_len_kurtosis'] = stats.kurtosis(lengths)
        else:
            features['pkt_len_skewness'] = 0
            features['pkt_len_kurtosis'] = 0
        
        # 分位数
        features['pkt_len_25th'] = np.percentile(lengths, 25)
        features['pkt_len_50th'] = np.percentile(lengths, 50)
        features['pkt_len_75th'] = np.percentile(lengths, 75)
        
        # 突发性指标
        if features['duration'] > 0:
            features['burstiness'] = features['std_iat'] / features['mean_iat'] if features['mean_iat'] > 0 else 0
            features['pkt_rate'] = features['total_packets'] / features['duration']
            features['byte_rate'] = features['total_bytes'] / features['duration']
        else:
            features['burstiness'] = 0
            features['pkt_rate'] = 0
            features['byte_rate'] = 0
        
        # 熵特征
        len_counts = Counter(lengths)
        len_probs = np.array(list(len_counts.values())) / len(lengths)
        features['pkt_len_entropy'] = -np.sum(len_probs * np.log2(len_probs + 1e-10))
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """返回空特征"""
        return {
            'total_packets': 0, 'total_bytes': 0,
            'min_pkt_len': 0, 'max_pkt_len': 0,
            'mean_pkt_len': 0, 'std_pkt_len': 0,
            'duration': 0,
            'mean_iat': 0, 'std_iat': 0, 'min_iat': 0, 'max_iat': 0, 'iat_cv': 0,
            'pkt_len_skewness': 0, 'pkt_len_kurtosis': 0,
            'pkt_len_25th': 0, 'pkt_len_50th': 0, 'pkt_len_75th': 0,
            'burstiness': 0, 'pkt_rate': 0, 'byte_rate': 0,
            'pkt_len_entropy': 0
        }


# ============================================================
# Layer 2: 时序行为特征
# ============================================================

class SequenceBehaviorFeatures:
    """时序行为特征提取器"""
    
    def __init__(self, max_len=100):
        self.max_len = max_len
    
    def extract(self, packets: List[Dict]) -> Dict[str, Any]:
        """提取时序行为特征"""
        if not packets:
            return self._get_empty_features()
        
        features = {}
        
        # 包大小序列
        lengths = [p['len'] for p in packets]
        features['length_sequence'] = self._pad_sequence(lengths)
        
        # IAT序列
        timestamps = [p['ts'] for p in packets]
        if len(timestamps) > 1:
            iats = np.diff(timestamps).tolist()
            features['iat_sequence'] = self._pad_sequence(iats)
        else:
            features['iat_sequence'] = np.zeros(self.max_len)
        
        # 序列趋势
        if len(lengths) > 2:
            x = np.arange(len(lengths))
            slope, _ = np.polyfit(x, lengths, 1)
            features['seq_trend'] = slope
        else:
            features['seq_trend'] = 0
        
        # 自相关系数
        if len(lengths) > 10:
            features['autocorr_lag1'] = self._autocorr(lengths, lag=1)
        else:
            features['autocorr_lag1'] = 0
        
        return features
    
    def _pad_sequence(self, seq: List) -> np.ndarray:
        """填充或截断序列到固定长度"""
        if len(seq) >= self.max_len:
            return np.array(seq[:self.max_len])
        else:
            return np.pad(seq, (0, self.max_len - len(seq)), 'constant')
    
    def _autocorr(self, x: List, lag: int = 1) -> float:
        """计算自相关系数"""
        try:
            c = np.correlate(x, x, mode='full')
            c = c[len(c)//2:]
            return c[lag] / c[0] if c[0] > 0 else 0
        except:
            return 0
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """返回空特征"""
        return {
            'length_sequence': np.zeros(self.max_len),
            'iat_sequence': np.zeros(self.max_len),
            'seq_trend': 0,
            'autocorr_lag1': 0
        }


# ============================================================
# Layer 3: 字节级序列特征（NetMamba风格）
# ============================================================

class ByteLevelFeatures:
    """字节级序列特征提取器"""
    
    def __init__(self, M=5, N_h=80, N_p=240):
        """
        M: 前M个包
        N_h: Header字节数
        N_p: Payload字节数
        """
        self.M = M
        self.N_h = N_h
        self.N_p = N_p
        self.total_bytes = M * (N_h + N_p)  # 5 * (80 + 240) = 1600
    
    def extract(self, packets: List[Dict]) -> Dict[str, Any]:
        """提取字节级特征"""
        features = {}
        
        # 构建1600字节序列
        byte_sequence = self._build_byte_sequence(packets)
        features['byte_sequence'] = byte_sequence
        
        # Stride切分 (1600字节 -> 400个stride，每个4字节)
        features['stride_sequence'] = self._create_strides(byte_sequence, stride_size=4)
        
        # 字节统计
        features['byte_entropy'] = self._calculate_entropy(byte_sequence)
        features['byte_uniformity'] = self._calculate_uniformity(byte_sequence)
        
        # Header解析（从第一个包）
        if packets and 'header' in packets[0]:
            header_features = self._parse_header(packets[0]['header'])
            features.update(header_features)
        else:
            features.update(self._get_empty_header_features())
        
        # Payload分析
        payload_bytes = self._extract_payload_bytes(packets)
        if len(payload_bytes) > 0:
            features['payload_entropy'] = self._calculate_entropy(payload_bytes)
            features['payload_ascii_ratio'] = np.sum((payload_bytes >= 32) & (payload_bytes <= 126)) / len(payload_bytes)
        else:
            features['payload_entropy'] = 0
            features['payload_ascii_ratio'] = 0
        
        return features
    
    def _build_byte_sequence(self, packets: List[Dict]) -> np.ndarray:
        """构建1600字节序列"""
        byte_seq = np.zeros(self.total_bytes, dtype=np.uint8)
        
        idx = 0
        for i, pkt in enumerate(packets[:self.M]):
            # Header
            if 'header' in pkt:
                header_bytes = self._hex_to_bytes(pkt['header'], self.N_h)
                byte_seq[idx:idx+self.N_h] = header_bytes
                idx += self.N_h
            else:
                idx += self.N_h
            
            # Payload
            if 'payload' in pkt:
                payload_bytes = self._hex_to_bytes(pkt['payload'], self.N_p)
                byte_seq[idx:idx+self.N_p] = payload_bytes
                idx += self.N_p
            else:
                idx += self.N_p
        
        return byte_seq
    
    def _hex_to_bytes(self, hex_str: str, max_len: int) -> np.ndarray:
        """16进制字符串转字节数组"""
        try:
            # 移除可能的空格
            hex_str = hex_str.replace(' ', '')
            # 转换为字节
            bytes_data = bytes.fromhex(hex_str)
            byte_array = np.frombuffer(bytes_data, dtype=np.uint8)
            # 截断或填充
            if len(byte_array) >= max_len:
                return byte_array[:max_len]
            else:
                return np.pad(byte_array, (0, max_len - len(byte_array)), 'constant')
        except:
            return np.zeros(max_len, dtype=np.uint8)
    
    def _create_strides(self, byte_seq: np.ndarray, stride_size: int = 4) -> np.ndarray:
        """创建stride序列"""
        num_strides = len(byte_seq) // stride_size
        strides = byte_seq[:num_strides * stride_size].reshape(num_strides, stride_size)
        return strides
    
    def _calculate_entropy(self, byte_array: np.ndarray) -> float:
        """计算字节熵"""
        if len(byte_array) == 0:
            return 0
        counts = np.bincount(byte_array, minlength=256)
        probs = counts[counts > 0] / len(byte_array)
        return -np.sum(probs * np.log2(probs))
    
    def _calculate_uniformity(self, byte_array: np.ndarray) -> float:
        """计算字节分布均匀度"""
        if len(byte_array) == 0:
            return 0
        counts = np.bincount(byte_array, minlength=256)
        probs = counts / len(byte_array)
        return 1 - np.sum((probs - 1/256)**2) * 256
    
    def _parse_header(self, header_hex: str) -> Dict[str, int]:
        """解析IP/TCP/UDP Header"""
        try:
            header_bytes = bytes.fromhex(header_hex.replace(' ', ''))
            if len(header_bytes) < 20:
                return self._get_empty_header_features()
            
            features = {}
            # IP Header (前20字节)
            features['ip_version'] = (header_bytes[0] >> 4) & 0x0F
            features['ip_protocol'] = header_bytes[9]
            features['ip_ttl'] = header_bytes[8]
            
            # TCP/UDP (从第20字节开始)
            if len(header_bytes) >= 40:
                if features['ip_protocol'] == 6:  # TCP
                    features['tcp_flags'] = header_bytes[33]
                    features['tcp_syn'] = int((features['tcp_flags'] & 0x02) > 0)
                    features['tcp_ack'] = int((features['tcp_flags'] & 0x10) > 0)
                    features['tcp_psh'] = int((features['tcp_flags'] & 0x08) > 0)
                    features['tcp_fin'] = int((features['tcp_flags'] & 0x01) > 0)
                else:
                    features['tcp_flags'] = 0
                    features['tcp_syn'] = 0
                    features['tcp_ack'] = 0
                    features['tcp_psh'] = 0
                    features['tcp_fin'] = 0
            else:
                features['tcp_flags'] = 0
                features['tcp_syn'] = 0
                features['tcp_ack'] = 0
                features['tcp_psh'] = 0
                features['tcp_fin'] = 0
            
            return features
        except:
            return self._get_empty_header_features()
    
    def _get_empty_header_features(self) -> Dict[str, int]:
        """空Header特征"""
        return {
            'ip_version': 0, 'ip_protocol': 0, 'ip_ttl': 0,
            'tcp_flags': 0, 'tcp_syn': 0, 'tcp_ack': 0, 'tcp_psh': 0, 'tcp_fin': 0
        }
    
    def _extract_payload_bytes(self, packets: List[Dict]) -> np.ndarray:
        """提取所有payload字节"""
        payload_bytes = []
        for pkt in packets[:self.M]:
            if 'payload' in pkt:
                try:
                    pb = bytes.fromhex(pkt['payload'].replace(' ', ''))
                    payload_bytes.extend(list(pb))
                except:
                    pass
        return np.array(payload_bytes, dtype=np.uint8) if payload_bytes else np.array([], dtype=np.uint8)


# ============================================================
# Layer 4: 深度语义特征
# ============================================================

class DeepSemanticFeatures:
    """深度语义特征提取器"""
    
    def extract(self, packets: List[Dict]) -> Dict[str, Any]:
        """提取深度语义特征"""
        if not packets:
            return self._get_empty_features()
        
        features = {}
        
        # 协议指纹检测
        features.update(self._detect_protocol_fingerprints(packets))
        
        # 加密检测
        features.update(self._detect_encryption(packets))
        
        # 字节转移图特征
        features.update(self._analyze_byte_transitions(packets))
        
        return features
    
    def _detect_protocol_fingerprints(self, packets: List[Dict]) -> Dict:
        """检测协议指纹"""
        features = {
            'has_tls_handshake': 0,
            'tls_version': 0,
            'has_http_method': 0,
            'has_user_agent': 0
        }
        
        for pkt in packets[:5]:
            if 'payload' not in pkt:
                continue
            
            try:
                payload_hex = pkt['payload'].replace(' ', '')
                payload_bytes = bytes.fromhex(payload_hex)
                
                # TLS检测 (0x16 = Handshake)
                if len(payload_bytes) > 5 and payload_bytes[0] == 0x16:
                    features['has_tls_handshake'] = 1
                    features['tls_version'] = (payload_bytes[1] << 8) | payload_bytes[2]
                
                # HTTP检测
                payload_str = payload_bytes[:100].decode('ascii', errors='ignore')
                if any(method in payload_str for method in ['GET ', 'POST ', 'PUT ', 'HEAD ']):
                    features['has_http_method'] = 1
                if 'User-Agent:' in payload_str:
                    features['has_user_agent'] = 1
            except:
                pass
        
        return features
    
    def _detect_encryption(self, packets: List[Dict]) -> Dict:
        """检测加密流量"""
        features = {}
        
        # 收集所有字节
        all_bytes = []
        for pkt in packets:
            if 'payload' in pkt:
                try:
                    pb = bytes.fromhex(pkt['payload'].replace(' ', ''))
                    all_bytes.extend(list(pb))
                except:
                    pass
        
        if len(all_bytes) > 0:
            byte_array = np.array(all_bytes, dtype=np.uint8)
            
            # 流熵
            counts = np.bincount(byte_array, minlength=256)
            probs = counts[counts > 0] / len(byte_array)
            features['flow_entropy'] = -np.sum(probs * np.log2(probs))
            
            # Chi-square test (均匀性检验)
            expected = len(byte_array) / 256
            chi2 = np.sum((counts - expected)**2 / expected)
            features['chi2_statistic'] = chi2
            features['chi2_p_value'] = 1 - stats.chi2.cdf(chi2, 255)
            
            # 字节均匀度
            features['byte_uniformity'] = 1 - np.sum((probs - 1/256)**2) * 256
            
            # 综合判断（熵>7.5 且 均匀度>0.6）
            features['is_likely_encrypted'] = int(
                features['flow_entropy'] > 7.5 and features['byte_uniformity'] > 0.6
            )
        else:
            features['flow_entropy'] = 0
            features['chi2_statistic'] = 0
            features['chi2_p_value'] = 1
            features['byte_uniformity'] = 0
            features['is_likely_encrypted'] = 0
        
        return features
    
    def _analyze_byte_transitions(self, packets: List[Dict]) -> Dict:
        """分析字节转移模式"""
        features = {}
        
        # 收集字节转移
        transitions = []
        for pkt in packets[:10]:
            if 'payload' not in pkt:
                continue
            try:
                pb = bytes.fromhex(pkt['payload'].replace(' ', ''))
                for i in range(len(pb)-1):
                    transitions.append((pb[i], pb[i+1]))
            except:
                pass
        
        if len(transitions) > 0:
            # 转移多样性
            features['transition_diversity'] = len(set(transitions))
            
            # 转移熵
            trans_counts = Counter(transitions)
            trans_probs = np.array(list(trans_counts.values())) / len(transitions)
            features['transition_entropy'] = -np.sum(trans_probs * np.log2(trans_probs + 1e-10))
        else:
            features['transition_diversity'] = 0
            features['transition_entropy'] = 0
        
        return features
    
    def _get_empty_features(self) -> Dict:
        """空特征"""
        return {
            'has_tls_handshake': 0, 'tls_version': 0,
            'has_http_method': 0, 'has_user_agent': 0,
            'flow_entropy': 0, 'chi2_statistic': 0, 'chi2_p_value': 1,
            'byte_uniformity': 0, 'is_likely_encrypted': 0,
            'transition_diversity': 0, 'transition_entropy': 0
        }


# ============================================================
# 主提取器
# ============================================================

class DeepFeatureExtractor:
    """深度特征提取器主类"""
    
    def __init__(self, layers=[1,2,3,4]):
        """
        layers: 要提取的层次列表
        1 = 增强统计
        2 = 时序行为
        3 = 字节级序列
        4 = 深度语义
        """
        self.layers = layers
        
        if 1 in layers:
            self.layer1 = EnhancedStatisticalFeatures()
        if 2 in layers:
            self.layer2 = SequenceBehaviorFeatures(max_len=100)
        if 3 in layers:
            self.layer3 = ByteLevelFeatures(M=5, N_h=80, N_p=240)
        if 4 in layers:
            self.layer4 = DeepSemanticFeatures()
    
    def extract_flow(self, flow_key, packets: List[Dict]) -> Dict:
        """提取单个流的所有特征"""
        features = {}
        
        # Flow ID
        features['flow_key'] = str(flow_key)
        
        # Layer 1
        if 1 in self.layers:
            features.update(self.layer1.extract(packets))
        
        # Layer 2
        if 2 in self.layers:
            features.update(self.layer2.extract(packets))
        
        # Layer 3
        if 3 in self.layers:
            features.update(self.layer3.extract(packets))
        
        # Layer 4
        if 4 in self.layers:
            features.update(self.layer4.extract(packets))
        
        return features
    
    def extract_all(self, flows_dict: Dict, max_flows=None) -> List[Dict]:
        """提取所有流的特征"""
        print(f"[+] 开始提取特征（Layers: {self.layers}）")
        
        all_features = []
        total = len(flows_dict)
        if max_flows:
            total = min(total, max_flows)
        
        for i, (flow_key, packets) in enumerate(flows_dict.items()):
            if max_flows and i >= max_flows:
                break
            
            if (i + 1) % 1000 == 0:
                print(f"  处理进度: {i+1}/{total}")
            
            features = self.extract_flow(flow_key, packets)
            all_features.append(features)
        
        print(f"[+] 特征提取完成！共 {len(all_features)} 条流")
        return all_features


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='深度特征提取器 - 四层金字塔')
    parser.add_argument('--flows', required=True, help='输入PKL文件（flows）')
    parser.add_argument('--out', required=True, help='输出文件（.pkl或.csv）')
    parser.add_argument('--layers', default='1,2,3,4', help='要提取的层次，逗号分隔（默认: 1,2,3,4）')
    parser.add_argument('--max-flows', type=int, help='最大处理流数（用于测试）')
    parser.add_argument('--format', choices=['pkl', 'csv', 'both'], default='both', 
                       help='输出格式（默认: both）')
    
    args = parser.parse_args()
    
    # 解析layers
    layers = [int(x) for x in args.layers.split(',')]
    print(f"[+] 将提取以下层次: {layers}")
    
    # 加载flows
    print(f"[+] 加载flows: {args.flows}")
    with open(args.flows, 'rb') as f:
        data = pickle.load(f)
    
    # 处理嵌套格式
    if isinstance(data, dict) and 'flows' in data:
        flows_dict = data['flows']
        print(f"[+] 检测到嵌套格式")
    else:
        flows_dict = data
    
    print(f"[+] 总流数: {len(flows_dict)}")
    
    # 提取特征
    extractor = DeepFeatureExtractor(layers=layers)
    features_list = extractor.extract_all(flows_dict, max_flows=args.max_flows)
    
    # 保存
    if args.format in ['pkl', 'both']:
        pkl_file = args.out if args.out.endswith('.pkl') else args.out.replace('.csv', '.pkl')
        print(f"[+] 保存PKL: {pkl_file}")
        with open(pkl_file, 'wb') as f:
            pickle.dump({'features': features_list}, f)
    
    if args.format in ['csv', 'both']:
        csv_file = args.out if args.out.endswith('.csv') else args.out.replace('.pkl', '.csv')
        print(f"[+] 保存CSV: {csv_file}")
        
        # 转换为DataFrame（排除数组类型字段）
        csv_features = []
        for feat in features_list:
            csv_feat = {}
            for k, v in feat.items():
                if isinstance(v, (int, float, str)):
                    csv_feat[k] = v
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    csv_feat[k] = float(v)
            csv_features.append(csv_feat)
        
        df = pd.DataFrame(csv_features)
        df.to_csv(csv_file, index=False)
        print(f"  CSV列数: {len(df.columns)}")
    
    print(f"\n[+] 完成！")
    print(f"  提取的流数: {len(features_list)}")
    print(f"  特征层次: {layers}")


if __name__ == '__main__':
    main()
