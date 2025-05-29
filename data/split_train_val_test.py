import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始csv
input_csv = '../data/train.csv'
df = pd.read_csv(input_csv)

# 先划分出test集（10%）
train_val, test = train_test_split(
    df, test_size=0.1, random_state=42, shuffle=True,
    stratify=df['label'] if 'label' in df.columns else None
)

# 再从剩下的中划分val集（10%）
train, val = train_test_split(
    train_val, test_size=0.1111, random_state=42, shuffle=True,
    stratify=train_val['label'] if 'label' in df.columns else None
)

# 保存
train.to_csv('../data/train_split.csv', index=False)
val.to_csv('../data/val_split.csv', index=False)
test.to_csv('../data/test_split.csv', index=False)

print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
