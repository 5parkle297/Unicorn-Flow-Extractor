# CICIDS2017 单向流特征提取工程

## 项目概述

本项目实现了从 CICIDS2017 数据集的 PCAP 文件中提取单向流（unidirectional flows）特征，用于网络流量分析和深度学习建模。

## 核心特性

- ✅ **单向流定义**: 使用 5-tuple + direction 作为流的唯一标识
- ✅ **Payload 提取**: 支持提取并截取前 N 字节的 payload（可配置）
- ✅ **TCP/UDP 支持**: 自动识别并处理两种协议
- ✅ **方向标识**: 基于 IP 字典序确定流方向
- ✅ **时间排序**: 每条流内的包按时间戳排序
- ✅ **编码兼容**: 自动处理 Windows tshark 的 UTF-16 LE 编码

## 文件说明

```
├── build_unidirectional_flows.py  # 主脚本：构建单向流
├── verify_flows.py                # 验证脚本：查看流统计和样例
├── extract_all.bat                # 批处理：提取所有 PCAP 文件
├── extract_packets_with_payload.md # 完整使用指南
└── README.md                      # 本文件
```

## 快速开始

### 步骤 1: 提取 Packet 数据（包含 Payload）

```bash
# 方式一：批量处理所有 PCAP 文件
extract_all.bat

# 方式二：单个文件处理
tshark -r Monday-WorkingHours.pcap -T fields -e frame.number -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e frame.len -e data.data -E header=y -E separator=, > packets-monday-full.csv
```

### 步骤 2: 构建单向流

```bash
python build_unidirectional_flows.py --csv packets-monday-full.csv --out flows-monday.pkl --payload-trim 128
```

**参数说明**:
- `--csv`: 输入的 packet CSV 文件（必需）
- `--out`: 输出的 pickle 文件（必需）
- `--payload-trim`: 截取 payload 前 N 字节，默认 128

### 步骤 3: 验证结果

```bash
python verify_flows.py flows-monday.pkl
```

## 输出格式

### 流数据结构

```python
flows = {
    (src_ip, dst_ip, src_port, dst_port, protocol, direction): [
        {
            'ts': 1499082958.598,      # 时间戳
            'len': 66,                  # 包长度
            'payload': '4500003c...'    # Payload hex 字符串
        },
        # ... 更多包
    ],
    # ... 更多流
}
```

### 流标识 (Key)

- `src_ip`, `dst_ip`: IP 地址字符串
- `src_port`, `dst_port`: 端口号（整数）
- `protocol`: 协议类型（6=TCP, 17=UDP）
- `direction`: 方向标识（0 或 1）
  - `0`: src_ip < dst_ip（字典序）
  - `1`: src_ip > dst_ip

## 组流逻辑详解

### 流标识 (Flow Key)

本项目使用 **六元组** (而非严格的五元组) 作为流的唯一标识:

```python
流标识 = (src_ip, dst_ip, src_port, dst_port, protocol, direction)
```

**组成要素**:
- `src_ip`, `dst_ip`: IP 地址字符串
- `src_port`, `dst_port`: 端口号(整数)
- `protocol`: 协议类型(6=TCP, 17=UDP)
- `direction`: 方向标识(0 或 1)

### 方向确定规则

方向基于 **IP 字典序** 确定:

```python
def determine_direction(ip1, ip2):
    """
    direction = 0 → ip1 < ip2 (字典序)
    direction = 1 → ip1 > ip2
    """
    return 0 if ip1 < ip2 else 1
```

**示例**:
- IP `192.168.1.100` < IP `8.8.8.8` → direction = 0
- IP `8.8.8.8` > IP `192.168.1.100` → direction = 1

### 流分割逻辑

#### 1. 基于方向分割

同一个 TCP 连接的两个方向被识别为 **两条独立的单向流**:

```
连接: A(192.168.1.1:1234) ↔ B(8.8.8.8:80)

单向流1 (A→B):
  key = ('192.168.1.1', '8.8.8.8', 1234, 80, 6, 0)

单向流2 (B→A):
  key = ('192.168.1.1', '8.8.8.8', 1234, 80, 6, 1)
```

#### 2. 基于超时分割 🆕

从 v2 版本开始,支持 **流超时机制**:

- 默认超时: **120 秒**
- 如果同一流超过 120 秒无活动,则自动分割为新流
- 新流标识: 在六元组基础上添加序号

```python
# 原流
key1 = ('192.168.1.1', '8.8.8.8', 1234, 80, 6, 0)

# 超时后的新流
key2 = ('192.168.1.1', '8.8.8.8', 1234, 80, 6, 0, 1)  # 序号=1
key3 = ('192.168.1.1', '8.8.8.8', 1234, 80, 6, 0, 2)  # 序号=2
```

**配置超时**:
```bash
# 启用120秒超时(默认)
python scripts/core/build_unidirectional_flows_v2.py --pcap input.pcap --out output.pkl

# 自定义超时时间
python scripts/core/build_unidirectional_flows_v2.py --pcap input.pcap --out output.pkl --flow-timeout 60

# 禁用超时
python scripts/core/build_unidirectional_flows_v2.py --pcap input.pcap --out output.pkl --no-timeout
```

### 为什么使用单向流?

**优势**:
1. ✅ **保留时序信息**: 包的到达顺序完整保留
2. ✅ **适合深度学习**: 每个方向独立建模
3. ✅ **实时检测**: 不需要等待双向通信完成
4. ✅ **隐私保护**: 可以只分析单方向流量

**应用场景**:
- 🎯 基于 Payload 的深度学习模型(LSTM/Transformer/Mamba)
- 🎯 实时入侵检测系统
- 🎯 非对称攻击检测(DDoS/端口扫描)
- 🎯 加密流量分析

详见: [md/UNIDIRECTIONAL_FLOW_RESEARCH.md](md/UNIDIRECTIONAL_FLOW_RESEARCH.md)

## 问题诊断与解决

### ✅ 问题 1: UnicodeDecodeError

**错误信息**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

**原因**: tshark 在 Windows 上默认输出 UTF-16 LE 编码的 CSV 文件

**解决方案**: 已在代码中修复，使用 `encoding="utf-16-le"`

### ⚠️ 问题 2: Payload 为空

**原因**: 当前 CSV 文件缺少 `data.data` 字段

**解决方案**: 
1. 使用 `extract_all.bat` 重新提取（已包含 `data.data` 字段）
2. 或手动运行完整的 tshark 命令（见快速开始）

### ℹ️ 问题 3: 流数量过多

对于一天的网络流量（几百万包），产生几十万条单向流是**正常**的。这是因为：
- 每个唯一的 5-tuple + direction 组合都会创建一个流
- 包含大量短连接（HTTP 请求、DNS 查询等）

## 统计信息示例

```
[+] Total flows: 262,917
[+] Total packets: 10,718,469
[+] Flow size stats:
    - Min packets: 1
    - Max packets: 1,389,757
    - Avg packets: 40.77
[+] Protocol distribution:
    - TCP flows: 262,917
    - UDP flows: 0
```

## 性能优化建议

### 内存优化

对于超大 PCAP 文件，可以：
1. 分批处理多个小文件
2. 使用流式处理而非一次性加载
3. 限制最大流数量或包数量

### 速度优化

1. **使用 SSD**: 大幅提升读写速度
2. **并行处理**: 同时处理多个 PCAP 文件
3. **减少 Payload 长度**: `--payload-trim 64` 而非 128

## 版本选择指南

本项目提供了多个版本的脚本,请根据需求选择:

### 流构建脚本

| 脚本 | 位置 | 功能 | 推荐场景 |
|------|------|------|----------|
| **build_unidirectional_flows_v2.py** ⭐ | `scripts/core/` | 完整版,支持PCAP/CSV,IP Masking,流超时,NetMamba格式 | **推荐使用** |
| build_unidirectional_flows.py | `archived/` | 旧版,仅支持CSV | 兼容旧工作流 |

**推荐**: 统一使用 **v2 版本**

**v2 新特性**:

- ✅ 支持直接解析 PCAP (无需 tshark)
- ✅ 支持 IP Masking (防止过拟合)
- ✅ 支持流超时机制 (防止长连接累积过多包)
- ✅ 支持 NetMamba 格式输出 (1600 字节数组)
- ✅ 更灵活的配置选项

### 特征提取脚本

| 脚本 | 位置 | 功能 | 推荐场景 |
|------|------|------|----------|
| **extract_features_deep.py** ⭐ | `scripts/features/` | 四层特征金字塔(统计+时序+字节+语义) | **深度学习研究** |
| extract_features_v2.py | `archived/` | 基础统计特征(~35维) | 传统机器学习 |

**选择建议**:

- 🎯 **深度学习研究**: 使用 `extract_features_deep.py` (信息保留 85%)
- 🎯 **传统机器学习**: 使用 `extract_features_v2.py` (快速,简洁)
- 🎯 **对比实验**: 两者都使用

详见: [md/DEEP_FEATURES_GUIDE.md](md/DEEP_FEATURES_GUIDE.md)

### 批处理脚本

| 脚本 | 位置 | 功能 | 执行位置 |
|------|------|------|----------|
| process_monday_1M.bat | `batch/` | 处理 Monday 1M 数据 | 在 `batch/` 目录执行 |
| quick_start.bat | `batch/` | 快速开始(适合初次使用) | 在项目根目录执行 |
| extract_all.bat | `batch/` | 批量提取 PCAP 到 CSV | 在包含 PCAP 的目录执行 |

**注意**: 批处理脚本已更新为使用 v2 版本

## 下一步工作

1. **流超时处理**: 添加超时机制（如 120 秒无活动则分割新流）
2. **统计特征提取**: 计算每条流的统计特征（字节数、持续时间等）
3. **标签整合**: 将 CICIDS2017 的攻击标签与流数据关联
4. **特征工程**: 将 payload hex 转换为模型输入格式

## 参考资料

- [CICIDS2017 数据集](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Wireshark/tshark 文档](https://www.wireshark.org/docs/man-pages/tshark.html)
- [单向流定义论文](相关论文链接)

## 常见问题 FAQ

### Q1: 为什么使用六元组而不是五元组?

**A**: 为了区分流的方向。传统五元组 `(src_ip, dst_ip, src_port, dst_port, protocol)` 无法区分 A→B 和 B→A,而单向流需要将它们视为两条不同的流。因此我们添加了 `direction` 字段。

### Q2: 流超时机制的作用是什么?

**A**: 防止长连接累积过多包。例如,一个 HTTP keep-alive 连接可能持续数小时,包含数千个请求。启用超时后,每 120 秒无活动会自动分割为新流,更符合"会话"的概念。

### Q3: 应该使用多大的超时时间?

**A**:

- **默认 120 秒**: 适合大多数场景
- **60 秒**: 更细粒度的流分割
- **300 秒**: 允许更长的空闲时间
- **禁用超时**: 用于研究长连接行为

### Q4: CSV 模式和 PCAP 模式有什么区别?

**A**:

- **CSV 模式**: 兼容现有工作流,需要先用 tshark 提取
- **PCAP 模式**: 直接解析 PCAP,可以精确分离 Header 和 Payload,支持 IP Masking

推荐使用 PCAP 模式(需要安装 scapy)。

### Q5: 为什么我的流数量很多?

**A**: 这是正常的。单向流会为每个唯一的六元组创建一条流。一天的网络流量可能包含:

- 数十万个短连接(HTTP 请求、DNS 查询)
- 每个连接的两个方向都是独立的流
- 启用超时后,长连接会被分割

### Q6: 旧版和新版脚本可以混用吗?

**A**: 可以,但不推荐。建议统一使用 v2 版本:

```bash
# 推荐
python scripts/core/build_unidirectional_flows_v2.py --pcap input.pcap --out output.pkl

# 不推荐混用
python archived/build_unidirectional_flows.py --csv packets.csv --out flows.pkl
```

### Q7: 如何选择特征提取脚本?

**A**:

- **深度学习**: 使用 `extract_features_deep.py` (四层特征,信息保留 85%)
- **传统 ML**: 使用 `extract_features_v2.py` (统计特征,快速)
- **对比研究**: 两者都使用,比较性能差异

### Q8: 批处理脚本报错怎么办?

**A**: 检查:

1. 是否在正确的目录执行(查看脚本注释)
2. Python 脚本路径是否正确
3. 输入文件是否存在
4. 依赖是否安装(`scapy`, `pandas`, `numpy`)

### Q9: 内存不足怎么办?

**A**:

- 使用 `--max-packets` 限制每流包数
- 分批处理大文件
- 启用流超时机制
- 增加系统内存

### Q10: 如何验证处理结果?

**A**: 使用验证脚本:

```bash
python scripts/utils/verify_flows.py data/flows/flows.pkl
```

查看:

- 流数量是否合理
- 包分布是否正常
- Payload 是否正确提取

## 许可证

本项目代码基于 MIT 许可证开源。

---

**最后更新**: 2025-11-28
**作者**: Network Traffic Analysis Team
