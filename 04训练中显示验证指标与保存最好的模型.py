# 导入所需的库
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import random
from tqdm import tqdm

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from AlexNet import AlexNet

from sklearn.metrics import confusion_matrix
import seaborn as sns

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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

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

model = AlexNet(num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Train_accuracy: {train_accuracy:.4f}%')

    model.eval()
    total, correct, test_loss, total_loss= 0, 0, 0, 0
    with torch.no_grad():
        # 使用tqdm包装train_loader
        progress_bar = tqdm(test_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss = criterion(outputs, labels)
            total_loss += test_loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_test_loss = total_loss / len(test_dataloader)
    acc = correct / total
    print(f"Test Data: Epoch [{epoch+1}/{num_epochs}], Loss {avg_test_loss:.4f}, Accuracy {acc * 100}%")
    if acc > most_acc:
        torch.save(model.state_dict(), f"D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\model\model_best.pth")
        most_acc = acc
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\model\models_{epoch+1}.pth")

# print("开始验证/评估模型：")
#
# model.load_state_dict(torch.load("./model/model_20.pth"))
# model.eval()
# total = 0
# correct = 0
# predicted_labels = []
# true_labels = []
# with torch.no_grad():
#     for images, labels in test_dataloader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         predicted_labels.extend(predicted.cpu().numpy())
#         true_labels.extend(labels.cpu().numpy())
#
# print(f"ACC {correct / total * 100}%")
#
# # 生成混淆矩阵
# conf = confusion_matrix(true_labels, predicted_labels)
# # 可视化
# sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("predict labels")
# plt.ylabel("true labels")
# plt.show()




