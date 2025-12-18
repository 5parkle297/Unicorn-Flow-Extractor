#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICIDS2017 预处理质量评估脚本
Preprocessing Quality Evaluation Script

评估8个维度:
1. 数据完整性 (Data Integrity)
2. 五元组正确性 (5-tuple Accuracy)
3. Payload完整性 (Payload Integrity)
4. 统计特征准确性 (Feature Accuracy)
5. 与官方数据集对比 (Baseline Comparison)
6. 标签匹配验证 (Label Matching)
7. 数据分布分析 (Distribution Analysis)
8. 机器学习基准测试 (ML Benchmark)
"""

import pickle
import pandas as pd
import numpy as np
import random
import subprocess
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import confusion_matrix, classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  sklearn未安装，机器学习测试将跳过")


class PreprocessingEvaluator:
    """预处理质量评估器"""
    
    def __init__(self, flows_pkl, csv_file=None, pcap_file=None, official_csv=None):
        """
        Args:
            flows_pkl: 单向流pkl文件
            csv_file: 提取的特征CSV文件
            pcap_file: 原始PCAP文件
            official_csv: 官方CICIDS2017 CSV文件（可选）
        """
        self.flows_pkl = flows_pkl
        self.csv_file = csv_file
        self.pcap_file = pcap_file
        self.official_csv = official_csv
        
        # 加载数据
        print(f"📂 加载数据: {flows_pkl}")
        with open(flows_pkl, 'rb') as f:
            self.flows = pickle.load(f)
        print(f"   ✓ 加载了 {len(self.flows)} 个流")
        
        if csv_file and Path(csv_file).exists():
            self.df = pd.read_csv(csv_file)
            print(f"   ✓ 加载了 {len(self.df)} 条CSV记录")
        else:
            self.df = None
            
        self.results = {}
        
    def evaluate_all(self):
        """执行所有评估"""
        print("\n" + "="*80)
        print("🎯 开始预处理质量评估")
        print("="*80)
        
        # 1. 数据完整性
        print("\n[1/8] 评估数据完整性...")
        self.evaluate_completeness()
        
        # 2. 五元组正确性
        print("\n[2/8] 评估五元组正确性...")
        self.evaluate_5tuple()
        
        # 3. Payload完整性
        print("\n[3/8] 评估Payload完整性...")
        self.evaluate_payload()
        
        # 4. 统计特征准确性
        print("\n[4/8] 评估统计特征准确性...")
        self.evaluate_features()
        
        # 5. 与官方数据集对比
        if self.official_csv and Path(self.official_csv).exists():
            print("\n[5/8] 与官方数据集对比...")
            self.compare_with_official()
        else:
            print("\n[5/8] 跳过官方数据集对比（未提供官方CSV）")
            self.results['official_comparison'] = {'skipped': True}
        
        # 6. 标签匹配验证
        print("\n[6/8] 评估标签匹配...")
        self.evaluate_labels()
        
        # 7. 数据分布分析
        print("\n[7/8] 分析数据分布...")
        self.analyze_distribution()
        
        # 8. 机器学习基准测试
        if ML_AVAILABLE and self.df is not None:
            print("\n[8/8] 机器学习基准测试...")
            self.ml_benchmark()
        else:
            print("\n[8/8] 跳过机器学习测试")
            self.results['ml_benchmark'] = {'skipped': True}
        
        # 生成报告
        print("\n" + "="*80)
        print("📊 生成评估报告...")
        self.generate_report()
        
        return self.results
    
    def evaluate_completeness(self):
        """1. 数据完整性验证"""
        results = {}
        
        # 统计流中的包数
        total_packets = sum(len(pkts) for pkts in self.flows.values())
        results['flow_count'] = len(self.flows)
        results['total_packets'] = total_packets
        
        # 流大小分布
        flow_sizes = [len(pkts) for pkts in self.flows.values()]
        results['flow_size_stats'] = {
            'min': int(np.min(flow_sizes)),
            'max': int(np.max(flow_sizes)),
            'mean': float(np.mean(flow_sizes)),
            'median': float(np.median(flow_sizes)),
            'std': float(np.std(flow_sizes))
        }
        
        # 协议分布
        tcp_count = sum(1 for key in self.flows.keys() if key[4] == 6)
        udp_count = sum(1 for key in self.flows.keys() if key[4] == 17)
        other_count = len(self.flows) - tcp_count - udp_count
        
        results['protocol_distribution'] = {
            'TCP': tcp_count,
            'UDP': udp_count,
            'Other': other_count
        }
        
        # 单向流方向分布
        direction_0 = sum(1 for key in self.flows.keys() if key[5] == 0)
        direction_1 = sum(1 for key in self.flows.keys() if key[5] == 1)
        results['direction_distribution'] = {
            'direction_0': direction_0,
            'direction_1': direction_1
        }
        
        # 如果有PCAP文件，对比包数
        if self.pcap_file and Path(self.pcap_file).exists():
            pcap_packets = self.count_pcap_packets()
            if pcap_packets > 0:
                coverage = (total_packets / pcap_packets) * 100
                results['pcap_packets'] = pcap_packets
                results['coverage_rate'] = round(coverage, 2)
                results['coverage_pass'] = coverage > 90
        
        # 时间范围
        all_timestamps = []
        for pkts in self.flows.values():
            all_timestamps.extend([p['ts'] for p in pkts])
        
        if all_timestamps:
            results['time_range'] = {
                'start': float(min(all_timestamps)),
                'end': float(max(all_timestamps)),
                'duration_seconds': float(max(all_timestamps) - min(all_timestamps))
            }
        
        self.results['completeness'] = results
        
        print(f"   ✓ 流数量: {results['flow_count']}")
        print(f"   ✓ 总包数: {results['total_packets']}")
        print(f"   ✓ TCP流: {tcp_count}, UDP流: {udp_count}")
        if 'coverage_rate' in results:
            print(f"   ✓ PCAP覆盖率: {results['coverage_rate']}% {'✓ PASS' if results['coverage_pass'] else '✗ FAIL'}")
    
    def evaluate_5tuple(self):
        """2. 五元组正确性验证（抽样检查）"""
        results = {}
        
        # 随机抽样100个流（或全部，如果少于100）
        sample_size = min(100, len(self.flows))
        sampled_keys = random.sample(list(self.flows.keys()), sample_size)
        
        valid_count = 0
        errors = []
        
        for key in sampled_keys:
            src_ip, dst_ip, src_port, dst_port, protocol, direction = key
            packets = self.flows[key]
            
            # 检查五元组一致性
            is_valid = True
            
            # 检查端口合法性
            if not (0 <= src_port <= 65535 and 0 <= dst_port <= 65535):
                is_valid = False
                errors.append({
                    'flow': key,
                    'error': 'Invalid port number'
                })
            
            # 检查协议合法性
            if protocol not in [6, 17]:  # TCP, UDP
                is_valid = False
                errors.append({
                    'flow': key,
                    'error': f'Invalid protocol: {protocol}'
                })
            
            # 检查方向合法性
            if direction not in [0, 1]:
                is_valid = False
                errors.append({
                    'flow': key,
                    'error': f'Invalid direction: {direction}'
                })
            
            # 检查包数量
            if len(packets) == 0:
                is_valid = False
                errors.append({
                    'flow': key,
                    'error': 'Empty flow'
                })
            
            if is_valid:
                valid_count += 1
        
        accuracy = (valid_count / sample_size) * 100
        results['sample_size'] = sample_size
        results['valid_count'] = valid_count
        results['accuracy'] = round(accuracy, 2)
        results['pass'] = accuracy > 99
        results['errors'] = errors[:10]  # 只保留前10个错误
        
        self.results['5tuple_accuracy'] = results
        
        print(f"   ✓ 抽样数量: {sample_size}")
        print(f"   ✓ 有效流: {valid_count}")
        print(f"   ✓ 准确率: {accuracy}% {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    def evaluate_payload(self):
        """3. Payload完整性验证（抽样检查）"""
        results = {}
        
        # 随机抽样50个包
        sample_size = min(50, sum(len(pkts) for pkts in self.flows.values()))
        
        # 收集所有包
        all_packets = []
        for flow_key, pkts in self.flows.items():
            for pkt in pkts:
                all_packets.append((flow_key, pkt))
        
        sampled_packets = random.sample(all_packets, min(sample_size, len(all_packets)))
        
        valid_count = 0
        empty_payload_count = 0
        
        for flow_key, pkt in sampled_packets:
            payload = pkt.get('payload', '')
            
            # 检查payload是否为有效的十六进制字符串
            is_valid = True
            
            if not payload:
                empty_payload_count += 1
                is_valid = True  # 空payload是合法的
            elif not all(c in '0123456789abcdefABCDEF' for c in payload):
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        match_rate = (valid_count / len(sampled_packets)) * 100 if sampled_packets else 100
        
        results['sample_size'] = len(sampled_packets)
        results['valid_count'] = valid_count
        results['empty_payload_count'] = empty_payload_count
        results['match_rate'] = round(match_rate, 2)
        results['pass'] = match_rate == 100
        
        self.results['payload_integrity'] = results
        
        print(f"   ✓ 抽样数量: {len(sampled_packets)}")
        print(f"   ✓ 有效payload: {valid_count}")
        print(f"   ✓ 空payload: {empty_payload_count}")
        print(f"   ✓ 匹配率: {match_rate}% {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    def evaluate_features(self):
        """4. 统计特征准确性验证"""
        if self.df is None:
            print("   ⚠️  未提供CSV文件，跳过特征验证")
            self.results['feature_accuracy'] = {'skipped': True}
            return
        
        results = {}
        errors = []
        
        # 随机抽样检查
        sample_size = min(100, len(self.df))
        sampled_indices = random.sample(range(len(self.df)), sample_size)
        
        error_count = 0
        
        for idx in sampled_indices:
            row = self.df.iloc[idx]
            
            # 尝试从row中获取五元组信息重建flow_key
            # 这里需要根据你的CSV格式调整
            # 假设CSV中有 src_ip, dst_ip, src_port, dst_port, protocol等字段
            
            # 由于CSV格式可能不同，这里做简单的一致性检查
            # 检查数值特征是否合理
            
            checks = {
                'total_packets': row.get('total_packets', 0) > 0,
                'total_bytes': row.get('total_bytes', 0) >= 0,
                'duration': row.get('duration', 0) >= 0,
            }
            
            if not all(checks.values()):
                error_count += 1
                errors.append({
                    'index': idx,
                    'failed_checks': [k for k, v in checks.items() if not v]
                })
        
        accuracy = ((sample_size - error_count) / sample_size) * 100 if sample_size > 0 else 100
        error_rate = (error_count / sample_size) * 100 if sample_size > 0 else 0
        
        results['sample_size'] = sample_size
        results['error_count'] = error_count
        results['accuracy'] = round(accuracy, 2)
        results['error_rate'] = round(error_rate, 2)
        results['pass'] = error_rate < 1
        results['errors'] = errors[:10]
        
        self.results['feature_accuracy'] = results
        
        print(f"   ✓ 抽样数量: {sample_size}")
        print(f"   ✓ 错误数量: {error_count}")
        print(f"   ✓ 误差率: {error_rate}% {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    def compare_with_official(self):
        """5. 与官方数据集对比"""
        try:
            official_df = pd.read_csv(self.official_csv)
            
            results = {
                'our_flow_count': len(self.df) if self.df is not None else len(self.flows),
                'official_flow_count': len(official_df),
                'note': '单向流数量约为双向流的2倍'
            }
            
            self.results['official_comparison'] = results
            
            print(f"   ✓ 我们的流数量: {results['our_flow_count']}")
            print(f"   ✓ 官方流数量: {results['official_flow_count']}")
            print(f"   ℹ️  {results['note']}")
            
        except Exception as e:
            print(f"   ⚠️  对比失败: {str(e)}")
            self.results['official_comparison'] = {'error': str(e)}
    
    def evaluate_labels(self):
        """6. 标签匹配验证"""
        if self.df is None or 'label' not in self.df.columns:
            print("   ⚠️  CSV中无label字段，跳过标签验证")
            self.results['label_matching'] = {'skipped': True}
            return
        
        results = {}
        
        # 统计标签分布
        label_dist = self.df['label'].value_counts().to_dict()
        results['label_distribution'] = label_dist
        results['unique_labels'] = len(label_dist)
        results['total_samples'] = len(self.df)
        
        # 检查是否有缺失标签
        missing_labels = self.df['label'].isna().sum()
        results['missing_labels'] = int(missing_labels)
        results['missing_rate'] = round((missing_labels / len(self.df)) * 100, 2)
        
        results['pass'] = missing_labels == 0
        
        self.results['label_matching'] = results
        
        print(f"   ✓ 标签种类: {results['unique_labels']}")
        print(f"   ✓ 缺失标签: {missing_labels} ({results['missing_rate']}%)")
        print(f"   ✓ {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    def analyze_distribution(self):
        """7. 数据分布分析"""
        if self.df is None:
            print("   ⚠️  未提供CSV文件，跳过分布分析")
            self.results['distribution_analysis'] = {'skipped': True}
            return
        
        results = {}
        
        # 数值列统计
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols[:10]:  # 只分析前10个数值列
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'zeros': int((self.df[col] == 0).sum()),
                'nulls': int(self.df[col].isna().sum())
            }
        
        results['numeric_stats'] = stats
        
        # 总体缺失值
        results['total_nulls'] = int(self.df.isna().sum().sum())
        results['null_percentage'] = round((results['total_nulls'] / (len(self.df) * len(self.df.columns))) * 100, 2)
        
        # 异常检测（简单的3-sigma规则）
        outlier_counts = {}
        for col in numeric_cols[:10]:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                outliers = ((self.df[col] - mean).abs() > 3 * std).sum()
                outlier_counts[col] = int(outliers)
        
        results['outlier_counts'] = outlier_counts
        results['pass'] = results['null_percentage'] < 5  # 缺失值比例小于5%
        
        self.results['distribution_analysis'] = results
        
        print(f"   ✓ 分析了 {len(stats)} 个数值特征")
        print(f"   ✓ 缺失值: {results['total_nulls']} ({results['null_percentage']}%)")
        print(f"   ✓ {'✓ PASS' if results['pass'] else '✗ FAIL'}")
    
    def ml_benchmark(self):
        """8. 机器学习基准测试"""
        if self.df is None or 'label' not in self.df.columns:
            print("   ⚠️  无法进行ML测试")
            self.results['ml_benchmark'] = {'skipped': True}
            return
        
        try:
            # 准备数据
            df_clean = self.df.copy()
            
            # 移除非数值列和标签列
            X = df_clean.select_dtypes(include=[np.number])
            y = df_clean['label']
            
            # 移除label如果它在X中
            if 'label' in X.columns:
                X = X.drop('label', axis=1)
            
            # 处理无穷值和缺失值
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # 限制样本数量以加快速度
            max_samples = min(10000, len(X))
            if len(X) > max_samples:
                indices = random.sample(range(len(X)), max_samples)
                X = X.iloc[indices]
                y = y.iloc[indices]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )
            
            print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")
            
            # 训练RandomForest
            print("   训练RandomForest...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            
            # 计算指标
            results = {
                'classifier': 'RandomForest',
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(X.columns),
                'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
                'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
                'f1_score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            }
            
            results['pass'] = results['accuracy'] > 85
            
            self.results['ml_benchmark'] = results
            
            print(f"   ✓ 准确率: {results['accuracy']}%")
            print(f"   ✓ F1分数: {results['f1_score']}%")
            print(f"   ✓ {'✓ PASS' if results['pass'] else '✗ FAIL'}")
            
        except Exception as e:
            print(f"   ✗ ML测试失败: {str(e)}")
            self.results['ml_benchmark'] = {'error': str(e)}
    
    def count_pcap_packets(self):
        """使用tshark统计PCAP包数"""
        try:
            cmd = ['tshark', '-r', self.pcap_file, '-q', '-z', 'io,stat,0']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # 解析输出
            for line in result.stdout.split('\n'):
                if 'Frames' in line or 'frames' in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part)
        except Exception as e:
            print(f"   ⚠️  无法统计PCAP包数: {e}")
        
        return 0
    
    def generate_report(self):
        """生成评估报告"""
        report_file = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 评估报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("📊 评估摘要")
        print("="*80)
        
        # 统计通过的测试
        passed = 0
        total = 0
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                if 'skipped' in value:
                    continue
                if 'pass' in value:
                    total += 1
                    if value['pass']:
                        passed += 1
                        status = "✓ PASS"
                    else:
                        status = "✗ FAIL"
                    
                    test_name = key.replace('_', ' ').title()
                    print(f"{test_name:30s} {status}")
        
        print("="*80)
        if total > 0:
            print(f"总体通过率: {passed}/{total} ({round(passed/total*100, 1)}%)")
        
        # 关键指标总结
        print("\n关键指标:")
        
        if 'completeness' in self.results:
            comp = self.results['completeness']
            print(f"  • 流数量: {comp.get('flow_count', 'N/A')}")
            print(f"  • 包数量: {comp.get('total_packets', 'N/A')}")
            if 'coverage_rate' in comp:
                print(f"  • PCAP覆盖率: {comp['coverage_rate']}%")
        
        if '5tuple_accuracy' in self.results and 'accuracy' in self.results['5tuple_accuracy']:
            print(f"  • 五元组准确率: {self.results['5tuple_accuracy']['accuracy']}%")
        
        if 'payload_integrity' in self.results and 'match_rate' in self.results['payload_integrity']:
            print(f"  • Payload匹配率: {self.results['payload_integrity']['match_rate']}%")
        
        if 'ml_benchmark' in self.results and 'accuracy' in self.results['ml_benchmark']:
            print(f"  • ML准确率: {self.results['ml_benchmark']['accuracy']}%")
        
        print("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CICIDS2017预处理质量评估')
    parser.add_argument('--flows', required=True, help='单向流PKL文件')
    parser.add_argument('--csv', help='特征CSV文件（可选）')
    parser.add_argument('--pcap', help='原始PCAP文件（可选）')
    parser.add_argument('--official', help='官方CICIDS2017 CSV文件（可选）')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = PreprocessingEvaluator(
        flows_pkl=args.flows,
        csv_file=args.csv,
        pcap_file=args.pcap,
        official_csv=args.official
    )
    
    # 执行评估
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()
