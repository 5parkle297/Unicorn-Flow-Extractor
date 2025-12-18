"""
单向流构建器 V2 - 整合 NFStream 和 NetMamba 的核心思路

核心改进:
1. IP Masking - 防止模型过拟合到特定IP
2. Header + Payload 分离 - 保留协议信息
3. Byte Balancing - 固定长度输入
4. 包数量限制 - 每流最多N个包
5. 智能过滤 - 可选过滤无payload包
6. 多种输出格式 - 兼容不同下游任务

支持两种输入模式:
1. CSV模式 - 使用tshark导出的CSV（兼容现有流程）
2. PCAP模式 - 直接解析PCAP文件（NetMamba风格，更完整）

作者: 基于NetMamba和NFStream思路整合
"""

import argparse
import csv
import pickle
import numpy as np
import binascii
from collections import defaultdict
from pathlib import Path
import sys

# 增加 CSV 字段大小限制
csv.field_size_limit(100 * 1024 * 1024)

# ==================== 配置参数 ====================

DEFAULT_CONFIG = {
    # 包数量限制
    'max_packets_per_flow': 100,     # 每流最大包数（NFStream风格）
    'netmamba_packets': 5,           # NetMamba使用5个包

    # Header和Payload长度（字节）
    'header_bytes': 80,              # NetMamba: 80字节header
    'payload_bytes': 240,            # NetMamba: 240字节payload

    # IP处理
    'ip_masking': True,              # 是否进行IP Masking
    'mask_ip': "0.0.0.0",            # 掩码后的IP地址

    # 过滤策略
    'filter_empty_payload': False,   # 是否过滤无payload的包

    # 🆕 流超时配置
    'flow_timeout': 120,             # 流超时时间(秒),120秒无活动则分割新流
    'enable_timeout': True,          # 是否启用超时机制

    # 输出格式
    'output_format': 'dict',         # dict, netmamba, sequence
}


# ==================== 工具函数 ====================

def safe_int(x):
    """安全转换为整数"""
    try:
        return int(x)
    except:
        return None


def safe_float(x):
    """安全转换为浮点数"""
    try:
        return float(x)
    except:
        return None


def hex_to_bytes(hex_str):
    """十六进制字符串转字节数组"""
    if not hex_str:
        return np.array([], dtype=np.uint8)
    try:
        return np.array([int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)], dtype=np.uint8)
    except:
        return np.array([], dtype=np.uint8)


def bytes_to_hex(byte_array):
    """字节数组转十六进制字符串"""
    return ''.join(f'{b:02x}' for b in byte_array)


def balance_bytes(data, target_length, pad_value=0):
    """
    Byte Balancing: 填充或截断到固定长度
    
    Args:
        data: 字节数组或十六进制字符串
        target_length: 目标字节长度
        pad_value: 填充值（默认0）
    
    Returns:
        固定长度的numpy数组
    """
    if isinstance(data, str):
        # 十六进制字符串
        hex_len = target_length * 2
        if len(data) > hex_len:
            data = data[:hex_len]
        else:
            data = data + '0' * (hex_len - len(data))
        return hex_to_bytes(data)
    else:
        # 字节数组
        data = np.array(data, dtype=np.uint8)
        if len(data) > target_length:
            return data[:target_length]
        else:
            padded = np.full(target_length, pad_value, dtype=np.uint8)
            padded[:len(data)] = data
            return padded


def determine_direction(ip1, ip2):
    """确定流方向（基于IP字典序）"""
    return 0 if str(ip1) < str(ip2) else 1


def should_split_flow(packets, new_pkt_ts, timeout=120):
    """
    判断是否需要因超时分割流

    Args:
        packets: 当前流的包列表
        new_pkt_ts: 新包的时间戳
        timeout: 超时时间(秒)

    Returns:
        bool: True表示需要分割
    """
    if not packets:
        return False

    last_pkt = packets[-1]
    return (new_pkt_ts - last_pkt['ts']) > timeout


# ==================== CSV模式处理 ====================

class CSVFlowBuilder:
    """基于tshark CSV的流构建器（兼容现有流程）"""
    
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
    
    def build_flows(self, csv_file):
        """
        从tshark CSV构建单向流

        Args:
            csv_file: tshark导出的CSV文件路径

        Returns:
            flows: dict, 流字典
            stats: dict, 统计信息
        """
        flows = defaultdict(list)
        flow_counters = defaultdict(int)  # 🆕 记录每个基础流的序号
        stats = {
            'total_packets': 0,
            'valid_packets': 0,
            'skipped_empty_payload': 0,
            'skipped_invalid': 0,
            'flows_split_by_timeout': 0  # 🆕 因超时分割的流数量
        }
        
        # 尝试不同编码
        encodings = ['utf-16-le', 'utf-8', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(csv_file, "r", newline='', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        stats['total_packets'] += 1
                        pkt = self._process_packet(row)
                        
                        if pkt is None:
                            stats['skipped_invalid'] += 1
                            continue
                        
                        # 可选：过滤无payload的包
                        if self.config['filter_empty_payload'] and not pkt['payload']:
                            stats['skipped_empty_payload'] += 1
                            continue

                        # 🆕 检查流超时
                        if self.config['enable_timeout']:
                            base_key = pkt['flow_key'][:5]  # 不含direction的五元组

                            # 检查是否需要因超时分割流
                            if pkt['flow_key'] in flows and flows[pkt['flow_key']]:
                                if should_split_flow(flows[pkt['flow_key']], pkt['ts'],
                                                    self.config['flow_timeout']):
                                    # 创建新的流key (添加流序号)
                                    flow_counters[base_key] += 1
                                    # 新key格式: (ip1, ip2, port1, port2, proto, direction, seq)
                                    new_key = (*pkt['flow_key'], flow_counters[base_key])
                                    pkt['flow_key'] = new_key
                                    stats['flows_split_by_timeout'] += 1

                        stats['valid_packets'] += 1
                        flows[pkt['flow_key']].append(pkt)
                    
                    break  # 成功读取，退出循环
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[!] Error reading CSV with {encoding}: {e}")
                continue
        
        # 后处理
        flows = self._post_process_flows(dict(flows))
        
        return flows, stats
    
    def _process_packet(self, row):
        """处理单个包"""
        # 提取基础字段
        src = row.get("ip.src", "")
        dst = row.get("ip.dst", "")
        
        sport = safe_int(row.get("tcp.srcport")) or safe_int(row.get("udp.srcport"))
        dport = safe_int(row.get("tcp.dstport")) or safe_int(row.get("udp.dstport"))
        
        if not src or not dst or sport is None or dport is None:
            return None
        
        # IP Masking（在流key中使用原始IP，但可以在特征中使用mask）
        original_src, original_dst = src, dst
        if self.config['ip_masking']:
            # 对于流分组，仍使用原始IP
            # 但在后续特征提取时可以mask
            pass
        
        # 协议判断
        if safe_int(row.get("tcp.srcport")) is not None:
            protocol = 6  # TCP
        elif safe_int(row.get("udp.srcport")) is not None:
            protocol = 17  # UDP
        else:
            protocol = -1
        
        # 方向和流key
        direction = determine_direction(src, dst)
        if direction == 0:
            flow_key = (src, dst, sport, dport, protocol, 0)
        else:
            flow_key = (dst, src, dport, sport, protocol, 1)
        
        # 时间戳和长度
        ts = safe_float(row.get("frame.time_epoch"))
        if ts is None:
            return None
        
        length = safe_int(row.get("frame.len")) or 0
        
        # Payload
        payload_hex = row.get("data.data", "") or ""
        # 清理payload（去除可能的空格和冒号）
        payload_hex = payload_hex.replace(" ", "").replace(":", "")
        
        # Byte Balancing for payload
        if payload_hex:
            payload_hex = payload_hex[:self.config['payload_bytes'] * 2]
        
        return {
            'flow_key': flow_key,
            'ts': ts,
            'len': length,
            'payload': payload_hex,
            'direction': direction,
            'original_src': original_src,
            'original_dst': original_dst,
            'src_port': sport,
            'dst_port': dport,
            'protocol': protocol
        }
    
    def _post_process_flows(self, flows):
        """流后处理"""
        processed = {}
        
        for key, packets in flows.items():
            # 按时间排序
            packets = sorted(packets, key=lambda x: x['ts'])
            
            # 限制包数量
            max_pkts = self.config['max_packets_per_flow']
            if len(packets) > max_pkts:
                packets = packets[:max_pkts]
            
            # 简化为最终格式
            processed[key] = [{
                'ts': p['ts'],
                'len': p['len'],
                'payload': p['payload'],
                'direction': p['direction']
            } for p in packets]
        
        return processed


# ==================== PCAP模式处理 ====================

class PcapFlowBuilder:
    """
    直接解析PCAP的流构建器（NetMamba风格）
    
    优势:
    - 可以提取完整的Header
    - 精确的Payload分离
    - 支持IP Masking
    """
    
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._check_scapy()
    
    def _check_scapy(self):
        """检查scapy是否可用"""
        try:
            import scapy.all as scapy
            self.scapy = scapy
            return True
        except ImportError:
            print("[!] Warning: scapy not installed. PCAP mode not available.")
            print("    Install with: pip install scapy")
            self.scapy = None
            return False
    
    def build_flows(self, pcap_file):
        """
        从PCAP构建单向流（NetMamba风格）

        Args:
            pcap_file: PCAP文件路径

        Returns:
            flows: dict, 流字典
            stats: dict, 统计信息
        """
        if self.scapy is None:
            raise ImportError("scapy is required for PCAP mode")

        flows = defaultdict(list)
        flow_counters = defaultdict(int)  # 🆕 记录每个基础流的序号
        stats = {
            'total_packets': 0,
            'valid_packets': 0,
            'skipped_no_ip': 0,
            'skipped_empty_payload': 0,
            'flows_split_by_timeout': 0  # 🆕 因超时分割的流数量
        }
        
        print(f"[+] Reading PCAP: {pcap_file}")
        packets = self.scapy.rdpcap(str(pcap_file))
        stats['total_packets'] = len(packets)
        
        for pkt in packets:
            result = self._process_packet(pkt)
            
            if result is None:
                stats['skipped_no_ip'] += 1
                continue
            
            if self.config['filter_empty_payload'] and not result['payload']:
                stats['skipped_empty_payload'] += 1
                continue

            # 🆕 检查流超时
            if self.config['enable_timeout']:
                base_key = result['flow_key'][:5]  # 不含direction的五元组

                # 检查是否需要因超时分割流
                if result['flow_key'] in flows and flows[result['flow_key']]:
                    if should_split_flow(flows[result['flow_key']], result['ts'],
                                        self.config['flow_timeout']):
                        # 创建新的流key (添加流序号)
                        flow_counters[base_key] += 1
                        # 新key格式: (ip1, ip2, port1, port2, proto, direction, seq)
                        new_key = (*result['flow_key'], flow_counters[base_key])
                        result['flow_key'] = new_key
                        stats['flows_split_by_timeout'] += 1

            stats['valid_packets'] += 1
            flows[result['flow_key']].append(result)
        
        # 后处理
        flows = self._post_process_flows(dict(flows))
        
        return flows, stats
    
    def _process_packet(self, pkt):
        """处理单个包（NetMamba风格）"""
        # 检查是否有IP层
        if not pkt.haslayer('IP'):
            return None
        
        ip_layer = pkt['IP']
        
        # 提取IP信息
        src = ip_layer.src
        dst = ip_layer.dst
        protocol = ip_layer.proto
        
        # 提取端口
        sport, dport = 0, 0
        if pkt.haslayer('TCP'):
            sport = pkt['TCP'].sport
            dport = pkt['TCP'].dport
        elif pkt.haslayer('UDP'):
            sport = pkt['UDP'].sport
            dport = pkt['UDP'].dport
        
        # 方向和流key（使用原始IP）
        direction = determine_direction(src, dst)
        if direction == 0:
            flow_key = (src, dst, sport, dport, protocol, 0)
        else:
            flow_key = (dst, src, dport, sport, protocol, 1)
        
        # 时间戳
        ts = float(pkt.time)
        
        # 包长度
        length = len(pkt)
        
        # 提取Header和Payload（NetMamba方式）
        header_hex, payload_hex = self._extract_header_payload(pkt)
        
        return {
            'flow_key': flow_key,
            'ts': ts,
            'len': length,
            'header': header_hex,
            'payload': payload_hex,
            'direction': direction,
            'original_src': src,
            'original_dst': dst,
            'src_port': sport,
            'dst_port': dport,
            'protocol': protocol
        }
    
    def _extract_header_payload(self, pkt):
        """
        提取Header和Payload（参考NetMamba的实现）
        
        Header: IP层字节（去除payload部分）
        Payload: Raw层字节
        """
        ip_layer = pkt['IP']
        
        # 对IP进行Masking（创建副本避免修改原始数据）
        if self.config['ip_masking']:
            ip_layer = ip_layer.copy()
            ip_layer.src = self.config['mask_ip']
            ip_layer.dst = self.config['mask_ip']
        
        # 提取header（整个IP包的字节）
        try:
            header_bytes = bytes(ip_layer)
            header_hex = binascii.hexlify(header_bytes).decode()
        except:
            header_hex = ''
        
        # 提取payload
        try:
            if pkt.haslayer('Raw'):
                payload_bytes = bytes(pkt['Raw'])
                payload_hex = binascii.hexlify(payload_bytes).decode()
                # 从header中移除payload
                header_hex = header_hex.replace(payload_hex, '')
            else:
                payload_hex = ''
        except:
            payload_hex = ''
        
        # Byte Balancing
        header_len = self.config['header_bytes'] * 2  # 十六进制字符数
        payload_len = self.config['payload_bytes'] * 2
        
        # Header: 截断或填充
        if len(header_hex) > header_len:
            header_hex = header_hex[:header_len]
        else:
            header_hex = header_hex + '0' * (header_len - len(header_hex))
        
        # Payload: 截断或填充
        if len(payload_hex) > payload_len:
            payload_hex = payload_hex[:payload_len]
        else:
            payload_hex = payload_hex + '0' * (payload_len - len(payload_hex))
        
        return header_hex, payload_hex
    
    def _post_process_flows(self, flows):
        """流后处理"""
        processed = {}
        
        for key, packets in flows.items():
            # 按时间排序
            packets = sorted(packets, key=lambda x: x['ts'])
            
            # 限制包数量
            max_pkts = self.config['max_packets_per_flow']
            if len(packets) > max_pkts:
                packets = packets[:max_pkts]
            
            # 保存为最终格式
            processed[key] = [{
                'ts': p['ts'],
                'len': p['len'],
                'header': p.get('header', ''),
                'payload': p.get('payload', ''),
                'direction': p['direction']
            } for p in packets]
        
        return processed


# ==================== NetMamba格式转换 ====================

def convert_to_netmamba_format(flows, config=None):
    """
    将流转换为NetMamba格式的numpy数组
    
    NetMamba格式:
    - 每个流: 5个包 × 320字节/包 = 1600字节
    - 每个包: 80字节header + 240字节payload
    
    Args:
        flows: 流字典
        config: 配置参数
    
    Returns:
        dict: {flow_key: numpy.array([1600,], dtype=uint8)}
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    netmamba_flows = {}
    
    num_packets = config['netmamba_packets']  # 5
    header_bytes = config['header_bytes']      # 80
    payload_bytes = config['payload_bytes']    # 240
    packet_bytes = header_bytes + payload_bytes  # 320
    flow_bytes = num_packets * packet_bytes      # 1600
    
    for key, packets in flows.items():
        # 取前5个包
        packets = packets[:num_packets]
        
        # 构建字节序列
        flow_hex_list = []
        
        for i in range(num_packets):
            if i < len(packets):
                pkt = packets[i]
                # 提取header和payload
                header = pkt.get('header', '0' * header_bytes * 2)
                payload = pkt.get('payload', '0' * payload_bytes * 2)
                
                # Balancing
                header = balance_bytes(header, header_bytes)
                payload = balance_bytes(payload, payload_bytes)
                
                # 连接
                pkt_bytes = np.concatenate([header, payload])
            else:
                # 不足5个包时用0填充
                pkt_bytes = np.zeros(packet_bytes, dtype=np.uint8)
            
            flow_hex_list.append(pkt_bytes)
        
        # 连接成一个数组
        flow_array = np.concatenate(flow_hex_list)
        assert len(flow_array) == flow_bytes, f"Expected {flow_bytes}, got {len(flow_array)}"
        
        netmamba_flows[key] = flow_array
    
    return netmamba_flows


# ==================== 主程序 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="单向流构建器 V2 - 整合NFStream和NetMamba思路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从tshark CSV构建流（兼容模式）
  python build_unidirectional_flows_v2.py --csv packets.csv --out flows.pkl
  
  # 从PCAP构建流（NetMamba模式）
  python build_unidirectional_flows_v2.py --pcap Monday.pcap --out flows.pkl --mode netmamba
  
  # 启用IP Masking
  python build_unidirectional_flows_v2.py --csv packets.csv --out flows.pkl --ip-masking
  
  # 限制每流5个包（NetMamba风格）
  python build_unidirectional_flows_v2.py --csv packets.csv --out flows.pkl --max-packets 5
  
  # 输出NetMamba格式（1600字节数组）
  python build_unidirectional_flows_v2.py --pcap Monday.pcap --out flows.pkl --format netmamba
        """
    )
    
    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="Input packet-level CSV from tshark")
    input_group.add_argument("--pcap", help="Input PCAP file (requires scapy)")
    
    # 输出
    parser.add_argument("--out", required=True, help="Output .pkl file")
    
    # 处理选项
    parser.add_argument("--max-packets", type=int, default=100,
                        help="Maximum packets per flow (default: 100)")
    parser.add_argument("--header-bytes", type=int, default=80,
                        help="Header length in bytes (default: 80)")
    parser.add_argument("--payload-bytes", type=int, default=240,
                        help="Payload length in bytes (default: 240)")
    
    # IP Masking
    parser.add_argument("--ip-masking", action="store_true",
                        help="Enable IP masking (set IP to 0.0.0.0)")
    parser.add_argument("--no-ip-masking", action="store_true",
                        help="Disable IP masking")
    
    # 过滤选项
    parser.add_argument("--filter-empty", action="store_true",
                        help="Filter packets without payload")

    # 🆕 流超时参数
    parser.add_argument("--flow-timeout", type=int, default=120,
                        help="Flow timeout in seconds. Split flow if idle > timeout (default: 120)")
    parser.add_argument("--no-timeout", action="store_true",
                        help="Disable flow timeout mechanism")

    # 输出格式
    parser.add_argument("--format", choices=['dict', 'netmamba', 'both'],
                        default='dict',
                        help="Output format: dict (default), netmamba (1600-byte arrays), both")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 构建配置
    config = {
        'max_packets_per_flow': args.max_packets,
        'header_bytes': args.header_bytes,
        'payload_bytes': args.payload_bytes,
        'ip_masking': args.ip_masking and not args.no_ip_masking,
        'filter_empty_payload': args.filter_empty,
        # 🆕 超时配置
        'flow_timeout': args.flow_timeout,
        'enable_timeout': not args.no_timeout,
    }
    
    print("=" * 60)
    print("单向流构建器 V2")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 最大包数/流: {config['max_packets_per_flow']}")
    print(f"  - Header长度: {config['header_bytes']} bytes")
    print(f"  - Payload长度: {config['payload_bytes']} bytes")
    print(f"  - IP Masking: {config['ip_masking']}")
    print(f"  - 过滤空payload: {config['filter_empty_payload']}")
    print(f"  - 流超时: {config['flow_timeout']}秒 (启用: {config['enable_timeout']})")
    print(f"  - 输出格式: {args.format}")
    print("=" * 60)
    
    # 选择处理模式
    if args.csv:
        print(f"\n[+] 模式: CSV (tshark)")
        print(f"[+] 输入: {args.csv}")
        builder = CSVFlowBuilder(config)
        flows, stats = builder.build_flows(args.csv)
    else:
        print(f"\n[+] 模式: PCAP (直接解析)")
        print(f"[+] 输入: {args.pcap}")
        builder = PcapFlowBuilder(config)
        flows, stats = builder.build_flows(args.pcap)
    
    # 打印统计
    print(f"\n[+] 统计:")
    print(f"  - 总包数: {stats['total_packets']:,}")
    print(f"  - 有效包: {stats['valid_packets']:,}")
    if 'skipped_invalid' in stats:
        print(f"  - 跳过(无效): {stats['skipped_invalid']:,}")
    if 'skipped_no_ip' in stats:
        print(f"  - 跳过(无IP): {stats['skipped_no_ip']:,}")
    if 'skipped_empty_payload' in stats:
        print(f"  - 跳过(无payload): {stats['skipped_empty_payload']:,}")
    if 'flows_split_by_timeout' in stats and stats['flows_split_by_timeout'] > 0:
        print(f"  - 因超时分割的流: {stats['flows_split_by_timeout']:,}")
    print(f"  - 总流数: {len(flows):,}")
    
    # 保存结果
    output_data = {'flows': flows, 'config': config, 'stats': stats}
    
    # 如果需要NetMamba格式
    if args.format in ['netmamba', 'both']:
        print(f"\n[+] 转换为NetMamba格式...")
        netmamba_flows = convert_to_netmamba_format(flows, config)
        output_data['netmamba_flows'] = netmamba_flows
        print(f"  - NetMamba流数: {len(netmamba_flows):,}")
        print(f"  - 每流字节数: 1600")
    
    print(f"\n[+] 保存到: {args.out}")
    with open(args.out, "wb") as f:
        pickle.dump(output_data, f)
    
    # 验证输出
    print(f"\n[+] 验证:")
    if flows:
        sample_key = list(flows.keys())[0]
        sample_flow = flows[sample_key]
        print(f"  - 示例流key: {sample_key}")
        print(f"  - 包数: {len(sample_flow)}")
        if sample_flow:
            print(f"  - 第一个包:")
            for k, v in sample_flow[0].items():
                if k == 'payload' and v:
                    print(f"      {k}: {v[:40]}... ({len(v)//2} bytes)")
                elif k == 'header' and v:
                    print(f"      {k}: {v[:40]}... ({len(v)//2} bytes)")
                else:
                    print(f"      {k}: {v}")
    
    print("\n[+] 完成!")


if __name__ == "__main__":
    main()
