"""
增强版流验证脚本 - 验证PKL文件与原始PCAP的对应关系

功能:
1. 基础统计验证 (流数量、包数量、协议分布)
2. PCAP包数量对应验证 (覆盖率检查)
3. 五元组准确性抽样验证 (随机抽样回溯PCAP)
4. 时间范围一致性验证
5. 生成验证报告

依赖: scapy, pickle
"""

import pickle
import random
import sys
from datetime import datetime
from collections import defaultdict

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("警告: scapy未安装,PCAP验证功能将被禁用")
    print("安装: pip install scapy")


def load_flows(pkl_file):
    """加载流数据"""
    print(f"[+] 加载流数据: {pkl_file}")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # 处理嵌套格式
    if isinstance(data, dict) and 'flows' in data:
        flows = data['flows']
        metadata = {k: v for k, v in data.items() if k != 'flows'}
    else:
        flows = data
        metadata = {}

    return flows, metadata


def basic_statistics(flows):
    """基础统计信息"""
    print("\n" + "="*80)
    print("【1】基础统计信息")
    print("="*80)

    total_flows = len(flows)
    total_packets = sum(len(pkts) for pkts in flows.values())

    print(f"✓ 总流数量: {total_flows:,}")
    print(f"✓ 总包数量: {total_packets:,}")
    print(f"✓ 平均每流包数: {total_packets/total_flows:.2f}")

    # 流大小分布
    flow_sizes = [len(pkts) for pkts in flows.values()]
    print(f"\n流大小分布:")
    print(f"  - 最小: {min(flow_sizes)} 包")
    print(f"  - 最大: {max(flow_sizes):,} 包")
    print(f"  - 中位数: {sorted(flow_sizes)[len(flow_sizes)//2]} 包")
    print(f"  - 平均: {sum(flow_sizes)/len(flow_sizes):.2f} 包")

    # 协议分布
    tcp_flows = sum(1 for key in flows.keys() if len(key) >= 5 and key[4] == 6)
    udp_flows = sum(1 for key in flows.keys() if len(key) >= 5 and key[4] == 17)
    other_flows = total_flows - tcp_flows - udp_flows

    print(f"\n协议分布:")
    print(f"  - TCP: {tcp_flows:,} ({tcp_flows/total_flows*100:.1f}%)")
    print(f"  - UDP: {udp_flows:,} ({udp_flows/total_flows*100:.1f}%)")
    if other_flows > 0:
        print(f"  - 其他: {other_flows:,} ({other_flows/total_flows*100:.1f}%)")

    # 方向分布
    dir0_flows = sum(1 for key in flows.keys() if len(key) >= 6 and key[5] == 0)
    dir1_flows = sum(1 for key in flows.keys() if len(key) >= 6 and key[5] == 1)

    print(f"\n方向分布:")
    print(f"  - Direction 0 (src<dst): {dir0_flows:,} ({dir0_flows/total_flows*100:.1f}%)")
    print(f"  - Direction 1 (src>dst): {dir1_flows:,} ({dir1_flows/total_flows*100:.1f}%)")

    # 超时分割统计
    timeout_flows = sum(1 for key in flows.keys() if len(key) > 6)
    if timeout_flows > 0:
        print(f"\n超时分割:")
        print(f"  - 因超时分割的流: {timeout_flows:,} ({timeout_flows/total_flows*100:.1f}%)")

    return {
        'total_flows': total_flows,
        'total_packets': total_packets,
        'tcp_flows': tcp_flows,
        'udp_flows': udp_flows,
        'flow_sizes': flow_sizes
    }


def verify_pcap_correspondence(flows, pcap_file):
    """验证PCAP对应关系"""
    if not SCAPY_AVAILABLE:
        print("\n⚠️  跳过PCAP验证 (scapy未安装)")
        return None

    print("\n" + "="*80)
    print("【2】PCAP对应关系验证")
    print("="*80)

    print(f"[+] 读取PCAP文件: {pcap_file}")
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"✗ 读取PCAP失败: {e}")
        return None

    pcap_packet_count = len(packets)
    flow_packet_count = sum(len(pkts) for pkts in flows.values())

    print(f"\n包数量对应:")
    print(f"  - PCAP包总数: {pcap_packet_count:,}")
    print(f"  - 流中包总数: {flow_packet_count:,}")

    # 计算覆盖率
    if pcap_packet_count > 0:
        coverage = flow_packet_count / pcap_packet_count * 100
        print(f"  - 覆盖率: {coverage:.2f}%")

        if coverage >= 90:
            print(f"  ✓ 覆盖率良好 (≥90%)")
        elif coverage >= 70:
            print(f"  ⚠️  覆盖率中等 (70-90%)")
        else:
            print(f"  ✗ 覆盖率偏低 (<70%)")

    # 时间范围验证
    pcap_times = [float(pkt.time) for pkt in packets if hasattr(pkt, 'time')]
    if pcap_times:
        pcap_start = min(pcap_times)
        pcap_end = max(pcap_times)

        flow_times = []
        for pkts in flows.values():
            for pkt in pkts:
                if 'ts' in pkt:
                    flow_times.append(pkt['ts'])

        if flow_times:
            flow_start = min(flow_times)
            flow_end = max(flow_times)

            print(f"\n时间范围验证:")
            print(f"  PCAP时间: {datetime.fromtimestamp(pcap_start)} ~ {datetime.fromtimestamp(pcap_end)}")
            print(f"  流时间:   {datetime.fromtimestamp(flow_start)} ~ {datetime.fromtimestamp(flow_end)}")

            if flow_start >= pcap_start and flow_end <= pcap_end:
                print(f"  ✓ 时间范围一致")
            else:
                print(f"  ⚠️  时间范围不一致")

    return {
        'pcap_packets': pcap_packet_count,
        'flow_packets': flow_packet_count,
        'coverage': coverage if pcap_packet_count > 0 else 0
    }


def verify_5tuple_sampling(flows, pcap_file, sample_size=20):
    """五元组抽样验证"""
    if not SCAPY_AVAILABLE:
        print("\n⚠️  跳过五元组验证 (scapy未安装)")
        return None

    print("\n" + "="*80)
    print("【3】五元组抽样验证")
    print("="*80)

    print(f"[+] 随机抽样 {sample_size} 个流进行验证...")

    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"✗ 读取PCAP失败: {e}")
        return None

    # 构建PCAP包索引 (五元组 -> 包列表)
    pcap_index = defaultdict(list)
    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            protocol = pkt[IP].proto

            src_port = dst_port = 0
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            # 使用方向0的格式存储
            direction = 0 if src_ip < dst_ip else 1
            key = (src_ip, dst_ip, src_port, dst_port, protocol, direction)
            pcap_index[key].append(pkt)

    # 随机抽样流
    sample_keys = random.sample(list(flows.keys()), min(sample_size, len(flows)))

    match_count = 0
    mismatch_details = []

    for i, flow_key in enumerate(sample_keys, 1):
        # 提取前6个元素 (忽略超时序号)
        base_key = flow_key[:6] if len(flow_key) >= 6 else flow_key

        flow_packets = flows[flow_key]
        pcap_packets = pcap_index.get(base_key, [])

        if len(pcap_packets) > 0:
            match_count += 1
            status = "✓"
        else:
            status = "✗"
            mismatch_details.append(flow_key)

        if i <= 5:  # 只显示前5个
            print(f"  [{i}] {status} Flow {base_key[:2]} - PKL:{len(flow_packets)}包, PCAP:{len(pcap_packets)}包")

    accuracy = match_count / sample_size * 100
    print(f"\n验证结果:")
    print(f"  - 抽样数量: {sample_size}")
    print(f"  - 匹配数量: {match_count}")
    print(f"  - 准确率: {accuracy:.1f}%")

    if accuracy >= 99:
        print(f"  ✓ 准确率优秀 (≥99%)")
    elif accuracy >= 95:
        print(f"  ⚠️  准确率良好 (95-99%)")
    else:
        print(f"  ✗ 准确率偏低 (<95%)")

    if mismatch_details:
        print(f"\n不匹配的流 (前5个):")
        for key in mismatch_details[:5]:
            print(f"    {key}")

    return {
        'sample_size': sample_size,
        'match_count': match_count,
        'accuracy': accuracy
    }


def generate_report(pkl_file, pcap_file, stats, pcap_verify, tuple_verify):
    """生成验证报告"""
    print("\n" + "="*80)
    print("【验证报告】")
    print("="*80)

    report = f"""
# 流数据验证报告

## 基本信息
- PKL文件: {pkl_file}
- PCAP文件: {pcap_file}
- 验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 基础统计
- 总流数量: {stats['total_flows']:,}
- 总包数量: {stats['total_packets']:,}
- TCP流: {stats['tcp_flows']:,} ({stats['tcp_flows']/stats['total_flows']*100:.1f}%)
- UDP流: {stats['udp_flows']:,} ({stats['udp_flows']/stats['total_flows']*100:.1f}%)

## 2. PCAP对应验证
"""

    if pcap_verify:
        report += f"""- PCAP包总数: {pcap_verify['pcap_packets']:,}
- 流中包总数: {pcap_verify['flow_packets']:,}
- 覆盖率: {pcap_verify['coverage']:.2f}%
- 状态: {'✓ 通过' if pcap_verify['coverage'] >= 90 else '⚠️ 需检查'}
"""
    else:
        report += "- 状态: ⚠️ 未执行 (scapy未安装)\n"

    report += "\n## 3. 五元组验证\n"

    if tuple_verify:
        report += f"""- 抽样数量: {tuple_verify['sample_size']}
- 匹配数量: {tuple_verify['match_count']}
- 准确率: {tuple_verify['accuracy']:.1f}%
- 状态: {'✓ 通过' if tuple_verify['accuracy'] >= 99 else '⚠️ 需检查'}
"""
    else:
        report += "- 状态: ⚠️ 未执行 (scapy未安装)\n"

    report += f"""
## 4. 综合评估

"""

    # 综合评分
    score = 0
    max_score = 3

    if stats['total_flows'] > 0 and stats['total_packets'] > 0:
        score += 1
        report += "✓ 数据完整性: 通过\n"
    else:
        report += "✗ 数据完整性: 失败\n"

    if pcap_verify and pcap_verify['coverage'] >= 90:
        score += 1
        report += "✓ PCAP对应: 通过\n"
    elif pcap_verify:
        report += "⚠️ PCAP对应: 需检查\n"
    else:
        report += "- PCAP对应: 未验证\n"
        max_score -= 1

    if tuple_verify and tuple_verify['accuracy'] >= 99:
        score += 1
        report += "✓ 五元组准确性: 通过\n"
    elif tuple_verify:
        report += "⚠️ 五元组准确性: 需检查\n"
    else:
        report += "- 五元组准确性: 未验证\n"
        max_score -= 1

    report += f"\n**总评: {score}/{max_score} 项通过**\n"

    if score == max_score:
        report += "\n✓ 数据质量优秀,可用于研究\n"
    elif score >= max_score * 0.7:
        report += "\n⚠️ 数据质量良好,建议进一步检查\n"
    else:
        report += "\n✗ 数据质量存在问题,需要修复\n"

    print(report)

    # 保存报告
    report_file = pkl_file.replace('.pkl', '_verification_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[+] 报告已保存: {report_file}")

    return report


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_flows_enhanced.py <flows.pkl> [pcap_file]")
        print("\n示例:")
        print("  python verify_flows_enhanced.py flows.pkl")
        print("  python verify_flows_enhanced.py flows.pkl Monday.pcap")
        sys.exit(1)

    pkl_file = sys.argv[1]
    pcap_file = sys.argv[2] if len(sys.argv) > 2 else None

    print("="*80)
    print("增强版流验证工具")
    print("="*80)

    # 加载流数据
    flows, metadata = load_flows(pkl_file)

    if metadata:
        print(f"\n元数据: {metadata}")

    # 1. 基础统计
    stats = basic_statistics(flows)

    # 2. PCAP对应验证
    pcap_verify = None
    if pcap_file:
        pcap_verify = verify_pcap_correspondence(flows, pcap_file)
    else:
        print("\n⚠️  未提供PCAP文件,跳过对应验证")

    # 3. 五元组抽样验证
    tuple_verify = None
    if pcap_file and SCAPY_AVAILABLE:
        tuple_verify = verify_5tuple_sampling(flows, pcap_file, sample_size=20)

    # 4. 生成报告
    if pcap_file:
        generate_report(pkl_file, pcap_file, stats, pcap_verify, tuple_verify)

    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)


if __name__ == "__main__":
    main()
