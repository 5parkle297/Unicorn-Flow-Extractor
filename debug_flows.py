import pickle
import sys

pkl_file = sys.argv[1] if len(sys.argv) > 1 else 'data/flows/flows_monday_1M_20251212_1741.pkl'

print(f"Loading: {pkl_file}")
with open(pkl_file, 'rb') as f:
    flows = pickle.load(f)

print(f"\nTotal flows: {len(flows)}")
print(f"Type: {type(flows)}")

# 查看第一个流的结构
first_key = list(flows.keys())[0]
print(f"\nFirst key: {first_key}")
print(f"Key type: {type(first_key)}")
print(f"Key length: {len(first_key)}")

first_value = flows[first_key]
print(f"\nFirst value type: {type(first_value)}")
print(f"First value length: {len(first_value)}")

if isinstance(first_value, (list, tuple)):
    print(f"\nFirst packet type: {type(first_value[0])}")
    print(f"First packet: {first_value[0]}")
    if isinstance(first_value[0], dict):
        print(f"First packet keys: {first_value[0].keys()}")
else:
    print(f"\nValue content (first 200 chars): {str(first_value)[:200]}")
