"""
端到端流水线：从 PCAP 到带标签的特征 CSV
完整自动化流程
"""

import subprocess
import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: PCAP -> Features with Labels"
    )
    parser.add_argument("--pcap", required=True, help="Input PCAP file")
    parser.add_argument("--labels", help="Optional: CICIDS2017 label CSV file")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    parser.add_argument("--payload-trim", type=int, default=128, 
                        help="Trim payload to N bytes (default: 128)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Extract bidirectional flow features")
    parser.add_argument("--skip-tshark", action="store_true",
                        help="Skip tshark extraction (use existing packet CSV)")
    parser.add_argument("--packet-csv", help="Existing packet CSV (if --skip-tshark)")
    return parser.parse_args()


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[OK] {description} completed successfully")
    return result


def main():
    args = parse_args()
    
    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # 文件名
    pcap_name = Path(args.pcap).stem
    packet_csv = args.packet_csv if args.skip_tshark else out_dir / f"{pcap_name}_packets.csv"
    flows_pkl = out_dir / f"{pcap_name}_flows.pkl"
    features_csv = out_dir / f"{pcap_name}_features.csv"
    final_csv = out_dir / f"{pcap_name}_final.csv"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         CICIDS2017 特征提取流水线                            ║
╚══════════════════════════════════════════════════════════════╝

配置:
  - PCAP 文件: {args.pcap}
  - 标签文件: {args.labels or '无 (将标记为 BENIGN)'}
  - 输出目录: {args.out_dir}
  - Payload 截取: {args.payload_trim} bytes
  - 双向流: {args.bidirectional}

输出文件:
  - Packet CSV: {packet_csv}
  - Flows PKL: {flows_pkl}
  - Features CSV: {features_csv}
  - Final CSV: {final_csv}
""")
    
    # ========== 步骤 1: tshark 提取 ==========
    if not args.skip_tshark:
        tshark_cmd = [
            "tshark",
            "-r", args.pcap,
            "-T", "fields",
            "-e", "frame.number",
            "-e", "frame.time_epoch",
            "-e", "frame.len",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "tcp.srcport",
            "-e", "tcp.dstport",
            "-e", "udp.srcport",
            "-e", "udp.dstport",
            "-e", "tcp.flags.fin",
            "-e", "tcp.flags.syn",
            "-e", "tcp.flags.rst",
            "-e", "tcp.flags.push",
            "-e", "tcp.flags.ack",
            "-e", "tcp.flags.urg",
            "-e", "data.data",
            "-E", "header=y",
            "-E", "separator=,"
        ]
        
        # 检查 tshark 是否可用
        try:
            subprocess.run(["tshark", "-v"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n[ERROR] tshark not found. Please install Wireshark.")
            print("        Or use --skip-tshark with --packet-csv if you already have packet CSV")
            sys.exit(1)
        
        print("\n[INFO] Extracting packets with tshark...")
        print("[INFO] This may take several minutes for large PCAP files...")
        
        with open(packet_csv, "w", encoding="utf-16-le") as f:
            result = subprocess.run(tshark_cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"\n[ERROR] tshark failed: {result.stderr}")
                sys.exit(1)
        
        print(f"\n[OK] Packets extracted to {packet_csv}")
    else:
        print(f"\n[INFO] Skipping tshark extraction, using {packet_csv}")
    
    # ========== 步骤 2: 构建单向流 ==========
    build_flow_cmd = [
        "python",
        "build_unidirectional_flows.py",
        "--csv", str(packet_csv),
        "--out", str(flows_pkl),
        "--payload-trim", str(args.payload_trim)
    ]
    
    run_command(build_flow_cmd, "构建单向流")
    
    # ========== 步骤 3: 提取特征 ==========
    extract_features_cmd = [
        "python",
        "extract_features.py",
        "--flows", str(flows_pkl),
        "--out", str(features_csv)
    ]
    
    if args.bidirectional:
        extract_features_cmd.append("--bidirectional")
    
    run_command(extract_features_cmd, "提取流特征")
    
    # ========== 步骤 4: 匹配标签 ==========
    if args.labels:
        match_labels_cmd = [
            "python",
            "match_labels.py",
            "--features", str(features_csv),
            "--labels", args.labels,
            "--out", str(final_csv),
            "--strategy", "fuzzy"
        ]
        
        run_command(match_labels_cmd, "匹配标签")
    else:
        print("\n[INFO] No label file provided, adding default BENIGN labels...")
        import pandas as pd
        df = pd.read_csv(features_csv)
        df['label'] = 'BENIGN'
        df.to_csv(final_csv, index=False)
        print(f"[OK] Saved to {final_csv}")
    
    # ========== 完成 ==========
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   流水线执行完成！                           ║
╚══════════════════════════════════════════════════════════════╝

最终输出文件: {final_csv}

下一步:
  1. 查看特征: head -20 {final_csv}
  2. 训练模型: python train_model.py --data {final_csv}
  3. 评估结果: python evaluate.py --data {final_csv}

""")


if __name__ == "__main__":
    main()
