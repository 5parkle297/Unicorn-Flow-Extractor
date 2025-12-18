import pandas as pd
import pickle
import numpy as np

# 读取CSV
df = pd.read_csv('data/features/features_monday_1M_all_layers_test.csv')
print('='*60)
print('CSV特征')
print('='*60)
print(f'行数: {len(df)}')
print(f'列数: {len(df.columns)}')
print(f'\n列名:')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')

# 读取PKL
with open('data/features/features_monday_1M_all_layers_test.pkl', 'rb') as f:
    data = pickle.load(f)

first_feat = data['features'][0]
print('\n' + '='*60)
print('PKL特征（第一个流）')
print('='*60)
print(f'总特征数: {len(first_feat)}')
print(f'\n特征keys:')
for i, key in enumerate(first_feat.keys(), 1):
    value = first_feat[key]
    if isinstance(value, np.ndarray):
        print(f'  {i:2d}. {key:30s} - 数组 {value.shape}')
    else:
        print(f'  {i:2d}. {key:30s} - {type(value).__name__}')

print('\n' + '='*60)
print('差异说明')
print('='*60)
print(f'CSV只有标量特征: {len(df.columns)}维')
print(f'PKL包含所有特征: {len(first_feat)}个（含数组）')
print(f'\nCSV缺少的特征（数组类型）:')
for key, value in first_feat.items():
    if isinstance(value, np.ndarray):
        print(f'  - {key:30s} shape={value.shape}')
