import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import v2
from tqdm import tqdm

# 读取 CSV（以 train_split.csv 为例）
csv_path = '../data/train.csv'
image_root = '../data'
df = pd.read_csv(csv_path)

# 缓存目录
bf_cache_dir = os.path.join(image_root, '/mnt/d/cache/BF')
fl_cache_dir = os.path.join(image_root, '/mnt/d/cache/FL')
os.makedirs(bf_cache_dir, exist_ok=True)
os.makedirs(fl_cache_dir, exist_ok=True)

# 定义 transform（与你 dataloader 中一致）
normalize_BF = v2.Normalize((0.5251, 0.5998, 0.6397), (0.2339, 0.1905, 0.1573))
normalize_FL = v2.Normalize((0.0804, 0.0489, 0.1264), (0.0732, 0.0579, 0.0822))

transform_BF = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize_BF,
])

transform_FL = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize_FL,
])

for name in tqdm(df['Name']):
    # 处理 BF 图像
    bf_path = os.path.join(image_root, 'BF/train', name)
    bf_img = Image.open(bf_path)
    bf_tensor = transform_BF(bf_img)
    torch.save(bf_tensor, os.path.join(bf_cache_dir, f"{name}.pt"))

    # 处理 FL 图像
    fl_path = os.path.join(image_root, 'FL/train', name)
    fl_img = Image.open(fl_path).convert("RGB")
    fl_tensor = transform_FL(fl_img)
    torch.save(fl_tensor, os.path.join(fl_cache_dir, f"{name}.pt"))
