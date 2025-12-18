@echo off
REM ============================================================
REM 自动整理 CICIDS2017 项目文件
REM ============================================================

echo.
echo ============================================================
echo   CICIDS2017 项目文件自动整理工具
echo ============================================================
echo.
echo 本脚本将：
echo   1. 创建新的目录结构
echo   2. 移动文件到对应目录
echo   3. 归档旧版本文件
echo   4. 更新批处理脚本路径
echo.
echo 警告：此操作会移动大量文件！
echo.
pause

REM ============================================================
REM 步骤 1: 创建目录结构
REM ============================================================
echo.
echo [1/5] 创建目录结构...

mkdir scripts 2>nul
mkdir scripts\core 2>nul
mkdir scripts\features 2>nul
mkdir scripts\evaluation 2>nul
mkdir scripts\utils 2>nul

mkdir batch 2>nul

mkdir data 2>nul
mkdir data\flows 2>nul
mkdir data\features 2>nul
mkdir data\results 2>nul

mkdir archived 2>nul

echo   ✓ 目录结构创建完成


REM ============================================================
REM 步骤 2: 移动 Python 脚本
REM ============================================================
echo.
echo [2/5] 移动 Python 脚本...

REM 核心处理
if exist build_unidirectional_flows_v2.py (
    move build_unidirectional_flows_v2.py scripts\core\ >nul
    echo   ✓ build_unidirectional_flows_v2.py -^> scripts\core\
)

REM 特征提取
if exist extract_features_deep.py (
    move extract_features_deep.py scripts\features\ >nul
    echo   ✓ extract_features_deep.py -^> scripts\features\
)

REM 评估与分析
if exist evaluate_preprocessing.py (
    move evaluate_preprocessing.py scripts\evaluation\ >nul
    echo   ✓ evaluate_preprocessing.py -^> scripts\evaluation\
)

if exist unsupervised_analysis.py (
    move unsupervised_analysis.py scripts\evaluation\ >nul
    echo   ✓ unsupervised_analysis.py -^> scripts\evaluation\
)

REM 工具脚本
if exist verify_flows.py (
    move verify_flows.py scripts\utils\ >nul
    echo   ✓ verify_flows.py -^> scripts\utils\
)

if exist test_flow.py (
    move test_flow.py scripts\utils\ >nul
    echo   ✓ test_flow.py -^> scripts\utils\
)

if exist match_labels.py (
    move match_labels.py scripts\utils\ >nul
    echo   ✓ match_labels.py -^> scripts\utils\
)

if exist NfstreamPlugin.py (
    move NfstreamPlugin.py scripts\utils\ >nul
    echo   ✓ NfstreamPlugin.py -^> scripts\utils\
)

REM 主管道
if exist run_pipeline.py (
    move run_pipeline.py scripts\ >nul
    echo   ✓ run_pipeline.py -^> scripts\
)


REM ============================================================
REM 步骤 3: 移动批处理脚本
REM ============================================================
echo.
echo [3/5] 移动批处理脚本...

if exist quick_start.bat (
    move quick_start.bat batch\ >nul
    echo   ✓ quick_start.bat -^> batch\
)

if exist extract_all.bat (
    move extract_all.bat batch\ >nul
    echo   ✓ extract_all.bat -^> batch\
)

if exist process_tuesday.bat (
    move process_tuesday.bat batch\ >nul
    echo   ✓ process_tuesday.bat -^> batch\
)

if exist test_deep_features.bat (
    move test_deep_features.bat batch\ >nul
    echo   ✓ test_deep_features.bat -^> batch\
)

if exist evaluate_test.bat (
    move evaluate_test.bat batch\ >nul
    echo   ✓ evaluate_test.bat -^> batch\
)


REM ============================================================
REM 步骤 4: 移动数据文件
REM ============================================================
echo.
echo [4/5] 移动数据文件...

REM PKL 流文件
if exist flows-monday.pkl (
    move flows-monday.pkl data\flows\ >nul
    echo   ✓ flows-monday.pkl -^> data\flows\
)

if exist flows-monday-v2.pkl (
    move flows-monday-v2.pkl data\flows\ >nul
    echo   ✓ flows-monday-v2.pkl -^> data\flows\
)

if exist flows-monday-1M.pkl (
    move flows-monday-1M.pkl data\flows\ >nul
    echo   ✓ flows-monday-1M.pkl -^> data\flows\
)

if exist flows-tuesday-1M.pkl (
    move flows-tuesday-1M.pkl data\flows\ >nul
    echo   ✓ flows-tuesday-1M.pkl -^> data\flows\
)

if exist flows-pcap-test.pkl (
    move flows-pcap-test.pkl data\flows\ >nul
    echo   ✓ flows-pcap-test.pkl -^> data\flows\
)

if exist test-layer2.pkl (
    move test-layer2.pkl data\flows\ >nul
    echo   ✓ test-layer2.pkl -^> data\flows\
)

REM CSV 特征文件
if exist features-test.csv (
    move features-test.csv data\features\ >nul
    echo   ✓ features-test.csv -^> data\features\
)

if exist features-tuesday.csv (
    move features-tuesday.csv data\features\ >nul
    echo   ✓ features-tuesday.csv -^> data\features\
)

if exist features-compact.csv (
    move features-compact.csv data\features\ >nul
    echo   ✓ features-compact.csv -^> data\features\
)

if exist test-layer1.csv (
    move test-layer1.csv data\features\ >nul
    echo   ✓ test-layer1.csv -^> data\features\
)

REM 分析结果
if exist analysis_results (
    move analysis_results data\results\ >nul
    echo   ✓ analysis_results\ -^> data\results\
)

if exist analysis_tuesday (
    move analysis_tuesday data\results\ >nul
    echo   ✓ analysis_tuesday\ -^> data\results\
)

REM 评估报告
if exist evaluation_report_*.json (
    move evaluation_report_*.json data\results\ >nul
    echo   ✓ evaluation_report_*.json -^> data\results\
)


REM ============================================================
REM 步骤 5: 归档旧版本文件
REM ============================================================
echo.
echo [5/5] 归档旧版本文件...

if exist extract_features.py (
    move extract_features.py archived\ >nul
    echo   ✓ extract_features.py -^> archived\ (已过时)
)

if exist extract_features_v2.py (
    move extract_features_v2.py archived\ >nul
    echo   ✓ extract_features_v2.py -^> archived\ (已被deep替代)
)

if exist build_unidirectional_flows.py (
    move build_unidirectional_flows.py archived\ >nul
    echo   ✓ build_unidirectional_flows.py -^> archived\ (已被v2替代)
)


REM ============================================================
REM 完成
REM ============================================================
echo.
echo ============================================================
echo   ✓ 文件整理完成！
echo ============================================================
echo.
echo 新的目录结构：
echo   scripts/         - Python脚本
echo   batch/           - 批处理脚本
echo   data/            - 数据文件
echo   archived/        - 旧版本归档
echo   pcapdata/        - PCAP文件
echo   pcapcsv/         - CSV数据
echo   md/              - 文档
echo   references/      - 参考文献
echo   NetMamba/        - NetMamba代码
echo.
echo 注意：批处理脚本中的路径可能需要手动更新！
echo 请查看 FILE_ORGANIZATION.md 了解详情
echo.
pause
