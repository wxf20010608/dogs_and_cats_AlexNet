# 导入所需的库
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import shutil
from pathlib import Path

def check_and_create_dataset_structure():
    # 定义基础路径和类别
    base_path = Path("/disk/AlexNet_dogs_and_cats/dataset")
    categories = ['dogs', 'cats']
    splits = ['train', 'test']
    
    # 检查并创建目录结构
    for split in splits:
        split_path = base_path / split
        print(f"\n检查 {split} 目录结构...")
        
        # 确保split目录存在
        split_path.mkdir(parents=True, exist_ok=True)
        
        # 检查每个类别文件夹
        for category in categories:
            category_path = split_path / category
            category_path.mkdir(exist_ok=True)
            
            # 统计有效图片数量
            valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
            valid_images = [f for f in category_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in valid_extensions]
            
            print(f"{category} 类别中有效图片数量: {len(valid_images)}")
            if len(valid_images) == 0:
                print(f"警告: {category} 文件夹中没有有效的图片文件!")

# check_and_create_dataset_structure()

# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")


transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # 随机仿射变换
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),#高斯模糊
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ]),
    "test": transforms.Compose([
        transforms.Resize(224), # 调整图片大小
        transforms.CenterCrop(224), # 中心裁剪图片
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化, 均值和标准差（RGB）
    ]),
}

train_dataset = datasets.ImageFolder(r"D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\dataset\train", transform=transform["train"])
test_dataset = datasets.ImageFolder(r"D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\dataset\test", transform=transform["test"])
print(len(train_dataset))
print(len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 修改显示图片的部分
examples = enumerate(test_dataloader)
batch_idx, (imgs, labels) = next(examples)
plt.figure(figsize=(10, 10))

for i in range(4):
    # 使用与数据加载器相同的均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 反归一化图像
    image = imgs[i].numpy()
    image = std[:, None, None] * image + mean[:, None, None]
    
    # 确保像素值在[0,1]范围内
    image = np.clip(image, 0, 1)
    
    # 转换通道顺序
    image = np.transpose(image, (1, 2, 0))
    
    plt.subplot(2, 2, i+1)
    plt.imshow(image)
    plt.title(f"Truth: {labels[i]}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()








