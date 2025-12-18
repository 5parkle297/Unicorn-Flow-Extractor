import pickle
import random
import sys

def verify_flows(pkl_file, num_samples=3):
    """验证并打印流样例"""
    
    print(f"[+] Loading flows from: {pkl_file}")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    # 处理嵌套格式
    if isinstance(data, dict) and 'flows' in data:
        flows = data['flows']
        print(f"[+] Data format: nested (with metadata)")
    else:
        flows = data
        print(f"[+] Data format: direct flows dict")
    
    print(f"[+] Total flows: {len(flows)}")
    print()
    
    # 统计信息
    total_packets = sum(len(pkts) for pkts in flows.values())
    print(f"[+] Total packets across all flows: {total_packets}")
    
    # 流大小分布
    flow_sizes = [len(pkts) for pkts in flows.values()]
    print(f"[+] Flow size stats:")
    print(f"    - Min packets: {min(flow_sizes)}")
    print(f"    - Max packets: {max(flow_sizes)}")
    print(f"    - Avg packets: {sum(flow_sizes)/len(flow_sizes):.2f}")
    print()
    
    # 协议分布
    tcp_flows = sum(1 for key in flows.keys() if key[4] == 6)
    udp_flows = sum(1 for key in flows.keys() if key[4] == 17)
    print(f"[+] Protocol distribution:")
    print(f"    - TCP flows: {tcp_flows}")
    print(f"    - UDP flows: {udp_flows}")
    print(f"    - Other: {len(flows) - tcp_flows - udp_flows}")
    print()
    
    # 随机采样展示
    print(f"[+] Random flow samples ({num_samples}):")
    print("="*80)
    
    sample_keys = random.sample(list(flows.keys()), min(num_samples, len(flows)))
    
    for i, key in enumerate(sample_keys, 1):
        # 兼容5元组和6元组格式
        if len(key) == 6:
            src_ip, dst_ip, src_port, dst_port, protocol, direction = key
            proto_name = "TCP" if protocol == 6 else "UDP" if protocol == 17 else f"Proto-{protocol}"
            dir_arrow = "→" if direction == 0 else "←"
            print(f"\n【Flow {i}】")
            print(f"  5-tuple: {src_ip}:{src_port} {dir_arrow} {dst_ip}:{dst_port} ({proto_name})")
            print(f"  Direction: {direction} ({'src < dst' if direction == 0 else 'src > dst'})")
        elif len(key) == 5:
            src_ip, dst_ip, src_port, dst_port, protocol = key
            proto_name = "TCP" if protocol == 6 else "UDP" if protocol == 17 else f"Proto-{protocol}"
            print(f"\n【Flow {i}】")
            print(f"  5-tuple: {src_ip}:{src_port} → {dst_ip}:{dst_port} ({proto_name})")
        else:
            print(f"\n【Flow {i}】")
            print(f"  Key format: {key}")
        
        print(f"  Total packets: {len(flows[key])}")
        
        # 展示前5个包
        print(f"  First 5 packets:")
        packets_list = list(flows[key]) if not isinstance(flows[key], list) else flows[key]
        for j, pkt in enumerate(packets_list[:5], 1):
            ts = pkt['ts']
            length = pkt['len']
            payload_preview = pkt['payload'][:40] + "..." if len(pkt['payload']) > 40 else pkt['payload']
            print(f"    [{j}] ts={ts:.6f}, len={length}, payload={payload_preview}")
        
        if len(packets_list) > 5:
            print(f"    ... (total {len(packets_list)} packets)")
    
    print("\n" + "="*80)
    print("[+] Verification complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_flows.py <flows.pkl>")
        sys.exit(1)
    
    verify_flows(sys.argv[1])
