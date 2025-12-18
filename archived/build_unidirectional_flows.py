import argparse
import csv
import pickle
import sys
from collections import defaultdict

# 增加 CSV 字段大小限制，避免 payload 字段过大导致错误
# 设置为 100MB，足够大的 payload 字段
csv.field_size_limit(100 * 1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build unidirectional flows from tshark CSV output"
    )
    parser.add_argument("--csv", required=True, help="Input packet-level CSV from tshark")
    parser.add_argument("--out", required=True, help="Output .pkl file containing flows dict")
    parser.add_argument("--payload-trim", type=int, default=128,
                        help="Trim payload to first N bytes (default=128)")
    return parser.parse_args()


def safe_int(x):
    """Convert string to int safely, empty string -> None."""
    try:
        return int(x)
    except:
        return None


def determine_direction(ip1, ip2):
    """
    定义单向流方向：
    direction = 0 → ip1 < ip2（字典序/字符串比较）
    direction = 1 → ip1 > ip2
    """
    return 0 if ip1 < ip2 else 1


def build_flows(csv_file, payload_trim=128):
    flows = defaultdict(list)

    # tshark 在 Windows 上默认输出 UTF-16 LE 编码的 CSV
    with open(csv_file, "r", newline='', encoding="utf-16-le") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # ------------------------
            # 1. 提取基础字段
            # ------------------------
            src = row.get("ip.src", "")
            dst = row.get("ip.dst", "")

            # port 兼容 TCP/UDP
            sport = safe_int(row.get("tcp.srcport")) or safe_int(row.get("udp.srcport"))
            dport = safe_int(row.get("tcp.dstport")) or safe_int(row.get("udp.dstport"))

            if not src or not dst or sport is None or dport is None:
                # 无效记录跳过
                continue

            # ------------------------
            # 2. 协议类型判断
            # ------------------------
            # 如果 CSV 没有 protocol 字段，使用 TCP/UDP 判断
            if safe_int(row.get("tcp.srcport")) is not None:
                protocol = 6       # TCP
            elif safe_int(row.get("udp.srcport")) is not None:
                protocol = 17      # UDP
            else:
                protocol = -1      # 其他协议（可忽略）

            # ------------------------
            # 3. direction + 5元组
            # ------------------------
            direction = determine_direction(src, dst)

            if direction == 0:
                key = (src, dst, sport, dport, protocol, 0)
            else:
                key = (dst, src, dport, sport, protocol, 1)

            # ------------------------
            # 4. 组织包信息
            # ------------------------
            ts = float(row["frame.time_epoch"])
            length = safe_int(row.get("frame.len")) or 0
            payload_hex = row.get("data.data", "")

            # trim payload 前 N 字节（payload_hex 是 hex 字符串，每 2 字符=1 byte）
            if payload_trim > 0 and payload_hex:
                payload_hex = payload_hex[:payload_trim * 2]

            pkt = {
                "ts": ts,
                "len": length,
                "payload": payload_hex
            }

            flows[key].append(pkt)

    # ------------------------
    # 5. 对每条流按时间排序
    # ------------------------
    for key in flows:
        flows[key] = sorted(flows[key], key=lambda x: x["ts"])

    return flows


def main():
    args = parse_args()

    print(f"[+] Loading packet CSV: {args.csv}")
    flows = build_flows(args.csv, payload_trim=args.payload_trim)

    print(f"[+] Total unidirectional flows: {len(flows)}")

    print(f"[+] Saving to {args.out} ...")
    with open(args.out, "wb") as f:
        pickle.dump(flows, f)

    print("[+] Done.")


if __name__ == "__main__":
    main()
