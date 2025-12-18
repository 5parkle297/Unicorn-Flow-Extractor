# CICIDS2017 数据处理指南

## 🎯 当前状态

✅ 已启动处理 Monday 1M 数据  
⏳ 正在运行中...

## 📁 目录结构

```
cicids2017/
├── original/              # 原始数据
│   ├── pcapdata/         # PCAP文件
│   └── pcapcsv/          # 官方CSV
├── data/                 # 处理后的数据
│   ├── flows/           # PKL流文件（带时间戳）
│   ├── features/        # CSV特征（带时间戳）
│   └── results/         # 分析结果
├── scripts/             # Python脚本
│   ├── core/           # 核心处理
│   ├── evaluation/     # 评估
│   └── utils/          # 工具
├── batch/               # 批处理脚本
└── archived/            # 旧版本
```

## 🚀 完整处理流程

### 方法1：手动执行（推荐，更灵活）

#### 步骤1: 构建单向流（PCAP → PKL）
```bash
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Monday-1M_00000_20170703195558.pcap \
    --out data/flows/flows_monday_1M_20251212_1741.pkl \
    --max-packets 100000
```

**参数说明：**
- `--pcap`: 输入PCAP文件
- `--out`: 输出PKL文件（带时间戳）
- `--max-packets`: 最大处理包数（100000约=1M）
- `--header-bytes`: Header字节数（默认80）
- `--payload-bytes`: Payload字节数（默认240）

**输出：**
- `data/flows/flows_monday_1M_YYYYMMDD_HHMM.pkl`

---

#### 步骤2: 提取特征（PKL → CSV）
```bash
python archived/extract_features_v2.py \
    --flows data/flows/flows_monday_1M_20251212_1741.pkl \
    --out data/features/features_monday_1M_20251212_1741.csv
```

**输出：**
- `data/features/features_monday_1M_YYYYMMDD_HHMM.csv`

---

#### 步骤3: 验证结果
```bash
python scripts/utils/verify_flows.py data/flows/flows_monday_1M_20251212_1741.pkl
```

---

#### 步骤4: 评估质量（可选）
```bash
python scripts/evaluation/evaluate_preprocessing.py \
    --flows data/flows/flows_monday_1M_20251212_1741.pkl \
    --csv data/features/features_monday_1M_20251212_1741.csv \
    --pcap original/pcapdata/Monday-1M_00000_20170703195558.pcap
```

---

### 方法2: 批处理脚本（需要修复路径）

```bash
cd batch
process_monday_1M.bat
```

**注意：** 批处理脚本路径需要更新！

---

## 📊 处理不同数据集

### Monday 1M 数据（12个分片）

```bash
# 分片1
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Monday-1M_00000_20170703195558.pcap \
    --out data/flows/flows_monday_1M_00000_$(date +%Y%m%d_%H%M).pkl \
    --max-packets 100000

# 分片2
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Monday-1M_00001_20170703201727.pcap \
    --out data/flows/flows_monday_1M_00001_$(date +%Y%m%d_%H%M).pkl \
    --max-packets 100000

# ... 以此类推
```

### Tuesday 1M 数据（12个分片）

```bash
# 分片1
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Tuesday-1M_00000_20170704195332.pcap \
    --out data/flows/flows_tuesday_1M_00000_$(date +%Y%m%d_%H%M).pkl \
    --max-packets 100000

# ... 以此类推
```

### 完整数据集

```bash
# Monday完整
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Monday-WorkingHours.pcap \
    --out data/flows/flows_monday_full_$(date +%Y%m%d_%H%M).pkl

# Tuesday完整
python scripts/core/build_unidirectional_flows_v2.py \
    --pcap original/pcapdata/Tuesday-WorkingHours.pcap \
    --out data/flows/flows_tuesday_full_$(date +%Y%m%d_%H%M).pkl

# 其他天...
```

---

## ⏱️ 预计处理时间

| 数据集 | 大小 | 预计时间 | 输出大小 |
|--------|------|----------|----------|
| Monday 1M (1个分片) | ~100MB | 2-5分钟 | ~10-50MB |
| Monday 1M (12个分片) | ~1.2GB | 30-60分钟 | ~100-500MB |
| Monday 完整 | ~11GB | 2-4小时 | ~1-5GB |

---

## 📝 文件命名规范

### PKL文件
```
flows_{dataset}_{subset}_{timestamp}.pkl

例如:
- flows_monday_1M_20251212_1741.pkl
- flows_tuesday_1M_00001_20251212_1800.pkl
- flows_monday_full_20251212_2000.pkl
```

### CSV文件
```
features_{dataset}_{subset}_{timestamp}.csv

例如:
- features_monday_1M_20251212_1741.csv
- features_tuesday_1M_00001_20251212_1800.csv
```

---

## 🔍 验证数据质量

### 快速验证
```bash
python scripts/utils/verify_flows.py data/flows/flows_monday_1M_20251212_1741.pkl
```

### 完整评估
```bash
python scripts/evaluation/evaluate_preprocessing.py \
    --flows data/flows/flows_monday_1M_20251212_1741.pkl \
    --csv data/features/features_monday_1M_20251212_1741.csv \
    --pcap original/pcapdata/Monday-1M_00000_20170703195558.pcap
```

**评估指标：**
- ✅ 数据完整性: 包覆盖率 > 90%
- ✅ 五元组正确性: 准确率 > 99%
- ✅ Payload完整性: 匹配率 100%
- ✅ 统计特征准确性: 误差率 < 1%

---

## ⚠️ 注意事项

### 1. 内存管理
- 1M数据：需要约2-4GB内存
- 完整数据：需要约16-32GB内存
- 建议分批处理大文件

### 2. 磁盘空间
- 确保至少有20GB可用空间
- PKL文件约为PCAP的10-50%
- CSV文件较小，约为PKL的5-10%

### 3. 处理顺序
建议按以下顺序处理：
1. Monday 1M（第1个分片）- 测试
2. 验证结果
3. Monday 1M（所有12个分片）
4. Tuesday 1M（所有12个分片）
5. 其他天的完整数据

### 4. 错误处理
如果遇到错误：
1. 检查Python依赖: `pip install scapy pandas numpy`
2. 检查PCAP文件路径
3. 查看错误日志
4. 减少`--max-packets`参数

---

## 📦 输出文件说明

### PKL文件结构
```python
{
    (src_ip, dst_ip, src_port, dst_port, protocol, direction): [
        {'ts': float, 'len': int, 'payload': str},
        {'ts': float, 'len': int, 'payload': str},
        ...
    ],
    ...
}
```

### CSV文件列
```
src_ip, dst_ip, src_port, dst_port, protocol, direction,
total_packets, total_bytes, duration, avg_pkt_size, ...
```

---

## 🎯 当前任务状态

- [x] 创建目录结构
- [x] 创建处理脚本
- [x] 启动处理 Monday 1M (分片00000)
- [ ] 等待处理完成（约2-5分钟）
- [ ] 提取特征
- [ ] 验证结果
- [ ] 处理其他分片

---

## 💡 快速参考

### 查看PKL内容
```bash
python scripts/utils/verify_flows.py data/flows/flows_monday_1M_20251212_1741.pkl
```

### 查看CSV内容
```bash
# Windows
type data\features\features_monday_1M_20251212_1741.csv | more

# 或用Python
python -c "import pandas as pd; print(pd.read_csv('data/features/features_monday_1M_20251212_1741.csv').head())"
```

### 统计数据
```python
import pickle
with open('data/flows/flows_monday_1M_20251212_1741.pkl', 'rb') as f:
    flows = pickle.load(f)
    print(f"流数量: {len(flows)}")
    print(f"包总数: {sum(len(pkts) for pkts in flows.values())}")
```

---

## 📚 相关文档

- `md/PREPROCESSING_EVALUATION.md` - 评估方案
- `md/EVALUATION_LOGIC.md` - 评估逻辑
- `FILE_ORGANIZATION.md` - 文件组织
- `README.md` - 项目说明

---

**当前时间：** 2025/12/12 17:42  
**当前状态：** ⏳ 正在处理 Monday 1M 00000
