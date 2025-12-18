import pickle
import os

pkl_file = "flows-monday.pkl"

# 检查文件是否存在
if os.path.exists(pkl_file):
    file_size = os.path.getsize(pkl_file)
    print(f"[+] File exists: {pkl_file}")
    print(f"[+] File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # 加载并显示基本信息
    print("\n[+] Loading flows...")
    with open(pkl_file, "rb") as f:
        flows = pickle.load(f)
    
    print(f"[+] Total flows: {len(flows):,}")
    
    # 统计信息
    total_packets = sum(len(pkts) for pkts in flows.values())
    print(f"[+] Total packets: {total_packets:,}")
    
    # 检查 payload
    has_payload_count = 0
    for key, packets in flows.items():
        for pkt in packets:
            if pkt.get('payload'):
                has_payload_count += 1
                break
    
    print(f"[+] Flows with payload: {has_payload_count:,}")
    
    # 显示一个示例流
    print("\n[+] Sample flow:")
    sample_key = list(flows.keys())[0]
    sample_packets = flows[sample_key]
    src_ip, dst_ip, src_port, dst_port, protocol, direction = sample_key
    
    print(f"  Key: {src_ip}:{src_port} -> {dst_ip}:{dst_port} (protocol={protocol}, dir={direction})")
    print(f"  Packets: {len(sample_packets)}")
    print(f"  First packet:")
    print(f"    ts: {sample_packets[0]['ts']}")
    print(f"    len: {sample_packets[0]['len']}")
    print(f"    payload (first 50 chars): {sample_packets[0]['payload'][:50]}...")
    
    print("\n[+] SUCCESS!")
else:
    print(f"[ERROR] File not found: {pkl_file}")
