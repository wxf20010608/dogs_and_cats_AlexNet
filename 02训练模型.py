# 导入所需的库
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from tqdm import tqdm

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from AlexNet import AlexNet

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

# 在setup_seed函数之后添加
def optimize_cuda_settings():
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # 启用 cudnn benchmark
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用 TF32
        torch.backends.cudnn.allow_tf32 = True
        # 设置GPU为确定性模式
        torch.cuda.deterministic = True
        # 清空GPU缓存
        torch.cuda.empty_cache()

optimize_cuda_settings()
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")


transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),  # 先调整到更大尺寸
        transforms.RandomCrop(224),     # 然后随机裁剪到224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
        transforms.RandomRotation(15),  # 增加旋转角度
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 修改DataLoader部分
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128,  # 增加batch_size
    shuffle=True,
    num_workers=4,   # 增加工作进程数
    pin_memory=True, # 使用固定内存
    prefetch_factor=2 # 预加载因子
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)


# # 打印一下图片
# examples = enumerate(test_dataloader)
# batch_idx, (imgs, labels) = next(examples)
# for i in range(4):
#     mean = np.array([0.5, 0.5, 0.5])
#     std = np.array([0.5, 0.5, 0.5])
#     image = imgs[i].numpy() * std[:, None, None] + mean[:, None, None]
#     # 将图片转成numpy数组，主要是转换通道和宽高位置
#     image = np.transpose(image, (1, 2, 0))
#     plt.subplot(2, 2, i+1)
#     plt.imshow(image)
#     plt.title(f"Truth: {labels[i]}")
# plt.show()

save_path = r'D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = AlexNet(num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
best_accuracy = 0.0  # 记录最佳准确率
best_model_path = os.path.join(save_path, 'best_model.pth')  # 最佳模型保存路径

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_loss = 0.0
    train_correct = 0.0
    train_total = 0
    # 使用tqdm包装train_loader
    progress_bar = tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]')
    for images, labels in progress_bar:
        # 将数据移动到设备上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新损失和进度条
        total_loss += loss.item()
        train_accuracy = 100 * train_correct / train_total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{train_accuracy:.2f}%'
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total

    # 验证阶段
    model.eval()
    val_correct = 0.0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        val_bar = tqdm(test_dataloader, desc=f'Validating')
        for images, labels in val_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            val_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{val_accuracy:.2f}%'
            })
    
    # 计算验证集平均损失和准确率
    avg_val_loss = val_loss / len(test_dataloader)
    val_accuracy = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'发现更好的模型，验证准确率: {val_accuracy:.2f}%，已保存至 {best_model_path}')
    
    # 每10个epoch保存一次检查点
    if (epoch+1) % 10 == 0:
        checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'检查点已保存至 {checkpoint_path}')

print(f'训练完成！最佳验证准确率: {best_accuracy:.2f}%')









