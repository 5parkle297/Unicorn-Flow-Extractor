# 文件组织说明

## 📂 当前根目录文件清单与用途

### 🐍 Python脚本（.py）

#### 核心处理脚本
| 文件 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `build_unidirectional_flows_v2.py` | **单向流构建**（最新版） | ✅ 保留 | 移至 scripts/core/ |
| `build_unidirectional_flows.py` | 单向流构建（旧版） | ⚠️ 已被v2替代 | 可删除或归档 |

#### 特征提取脚本
| 文件 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `extract_features_deep.py` | **深度特征提取**（4层金字塔，最新） | ✅ 保留 | 移至 scripts/features/ |
| `extract_features_v2.py` | 特征提取v2（基础统计） | ⚠️ 已被deep替代 | 可删除或归档 |
| `extract_features.py` | 特征提取v1（最旧） | ❌ 已过时 | 删除 |

#### 评估与验证脚本
| 文件 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `evaluate_preprocessing.py` | **预处理质量评估**（8维度评估） | ✅ 保留 | 移至 scripts/evaluation/ |
| `verify_flows.py` | 流验证工具 | ✅ 保留 | 移至 scripts/utils/ |
| `test_flow.py` | 流测试 | ✅ 保留 | 移至 scripts/utils/ |
| `unsupervised_analysis.py` | 无监督分析 | ✅ 保留 | 移至 scripts/evaluation/ |

#### 其他工具脚本
| 文件 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `match_labels.py` | 标签匹配 | ✅ 保留 | 移至 scripts/utils/ |
| `run_pipeline.py` | 运行管道 | ✅ 保留 | 移至 scripts/ |
| `NfstreamPlugin.py` | Nfstream插件 | ✅ 保留 | 移至 scripts/utils/ |

---

### 💾 批处理脚本（.bat）

| 文件 | 用途 | 状态 | 建议 |
|------|------|------|------|
| `quick_start.bat` | **快速开始**（推荐入口） | ✅ 保留 | 移至 batch/ |
| `extract_all.bat` | 批量提取所有数据 | ✅ 保留 | 移至 batch/ |
| `process_tuesday.bat` | 处理周二数据 | ✅ 保留 | 移至 batch/ |
| `test_deep_features.bat` | 测试深度特征 | ✅ 保留 | 移至 batch/ |
| `evaluate_test.bat` | **评估测试**（新建） | ✅ 保留 | 移至 batch/ |

---

### 📦 PKL文件（.pkl）- 流数据

| 文件 | 用途 | 大小估计 | 建议 |
|------|------|---------|------|
| `flows-monday.pkl` | Monday完整流数据 | 大 | 移至 data/flows/ |
| `flows-monday-v2.pkl` | Monday v2版本 | 大 | 根据需要保留一个 |
| `flows-monday-1M.pkl` | Monday采样（1M包） | 中 | 移至 data/flows/ |
| `flows-tuesday-1M.pkl` | Tuesday采样（1M包） | 中 | 移至 data/flows/ |
| `flows-pcap-test.pkl` | PCAP测试数据 | 小 | 移至 data/flows/ |
| `test-layer2.pkl` | Layer2测试数据 | 小 | 移至 data/flows/ |

---

### 📊 CSV文件（.csv）- 特征数据

| 文件 | 用途 | 建议 |
|------|------|------|
| `features-test.csv` | 测试特征 | 移至 data/features/ |
| `features-tuesday.csv` | Tuesday特征 | 移至 data/features/ |
| `features-compact.csv` | 紧凑特征 | 移至 data/features/ |
| `test-layer1.csv` | Layer1测试 | 移至 data/features/ |

---

### 📁 已存在的目录

| 目录 | 用途 | 状态 |
|------|------|------|
| `pcapdata/` | 原始PCAP文件 | ✅ 保留 |
| `pcapcsv/` | PCAP导出的CSV | ✅ 保留 |
| `md/` | 所有文档 | ✅ 保留 |
| `references/` | 参考文献 | ✅ 保留 |
| `NetMamba/` | NetMamba代码 | ✅ 保留 |
| `analysis_results/` | 分析结果 | ✅ 保留，移至 data/results/ |
| `analysis_tuesday/` | Tuesday分析 | ✅ 保留，移至 data/results/ |

---

## 🗂️ 建议的新目录结构

```
cicids2017/
├── README.md                      # 项目说明
│
├── scripts/                       # 所有Python脚本
│   ├── core/                      # 核心处理
│   │   └── build_unidirectional_flows_v2.py
│   ├── features/                  # 特征提取
│   │   └── extract_features_deep.py
│   ├── evaluation/                # 评估与分析
│   │   ├── evaluate_preprocessing.py
│   │   └── unsupervised_analysis.py
│   ├── utils/                     # 工具脚本
│   │   ├── verify_flows.py
│   │   ├── test_flow.py
│   │   ├── match_labels.py
│   │   └── NfstreamPlugin.py
│   └── run_pipeline.py            # 主管道
│
├── batch/                         # 批处理脚本
│   ├── quick_start.bat
│   ├── extract_all.bat
│   ├── process_tuesday.bat
│   ├── test_deep_features.bat
│   └── evaluate_test.bat
│
├── data/                          # 所有数据文件
│   ├── flows/                     # PKL流文件
│   │   ├── flows-monday.pkl
│   │   ├── flows-monday-1M.pkl
│   │   ├── flows-tuesday-1M.pkl
│   │   └── ...
│   ├── features/                  # CSV特征文件
│   │   ├── features-test.csv
│   │   ├── features-tuesday.csv
│   │   └── ...
│   └── results/                   # 分析结果
│       ├── analysis_results/
│       ├── analysis_tuesday/
│       └── evaluation_report_*.json
│
├── archived/                      # 归档文件（旧版本）
│   ├── build_unidirectional_flows.py
│   ├── extract_features_v2.py
│   ├── extract_features.py
│   └── flows-monday-v2.pkl
│
├── pcapdata/                      # 原始PCAP（保持不变）
├── pcapcsv/                       # PCAP的CSV（保持不变）
├── md/                            # 文档（保持不变）
├── references/                    # 参考文献（保持不变）
└── NetMamba/                      # NetMamba代码（保持不变）
```

---

## 🗑️ 建议删除的文件

### 可以安全删除（已被新版本替代）
1. `extract_features.py` - 最旧版本，已被v2和deep替代
2. `build_unidirectional_flows.py` - 如果v2版本稳定

### 建议归档（保留备份但移出主目录）
1. `extract_features_v2.py` - 被deep替代，但可能还有参考价值
2. `flows-monday-v2.pkl` - 如果有flows-monday.pkl

---

## 📝 文件版本说明

### 单向流构建
- ❌ `build_unidirectional_flows.py` - v1（旧版）
- ✅ `build_unidirectional_flows_v2.py` - v2（当前使用）

### 特征提取
- ❌ `extract_features.py` - v1（最旧，删除）
- ⚠️ `extract_features_v2.py` - v2（基础统计，归档）
- ✅ `extract_features_deep.py` - v3（4层金字塔，当前最佳）

### 流数据
- `flows-monday.pkl` vs `flows-monday-v2.pkl` - 保留一个即可
- `flows-*-1M.pkl` - 采样数据，用于快速测试

---

## 🎯 推荐的清理步骤

### 步骤1：创建新目录结构
```bash
mkdir scripts scripts/core scripts/features scripts/evaluation scripts/utils
mkdir batch
mkdir data data/flows data/features data/results
mkdir archived
```

### 步骤2：移动文件
```bash
# Python脚本
move scripts/core/
move scripts/features/
move scripts/evaluation/
move scripts/utils/

# 批处理
move *.bat batch/

# 数据文件
move *.pkl data/flows/
move *.csv data/features/
move analysis_* data/results/
```

### 步骤3：归档旧文件
```bash
move extract_features.py archived/
move extract_features_v2.py archived/
move build_unidirectional_flows.py archived/
```

### 步骤4：更新批处理脚本路径
需要更新所有.bat文件中的Python脚本路径

---

## ✅ 清理后的根目录（简洁版）

```
cicids2017/
├── README.md
├── scripts/          ← Python脚本
├── batch/            ← 批处理脚本
├── data/             ← 数据文件
├── archived/         ← 旧版本归档
├── pcapdata/         ← PCAP文件
├── pcapcsv/          ← CSV数据
├── md/               ← 文档
├── references/       ← 参考文献
└── NetMamba/         ← NetMamba代码
```

根目录只剩下 README.md 和9个文件夹，清爽！

---

## 💡 注意事项

1. **备份重要数据**：移动前先备份 flows.pkl 和 features.csv
2. **更新路径**：移动后需要更新批处理脚本中的相对路径
3. **测试验证**：移动后运行测试确保所有功能正常
4. **Git管理**：如果使用Git，注意.gitignore排除大文件

---

## 🚀 快速清理命令（Windows）

我可以为你生成一个自动整理脚本 `organize_files.bat`
