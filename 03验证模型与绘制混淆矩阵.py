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


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

if __name__ == '__main__':
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

    model = AlexNet(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optomizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epoches = 100
    # for epoch in range(epoches):
    #     model.train()
    #     total_loss = 0
    #     for i, (images, labels) in enumerate(train_dataloader):
    #         # 数据放在设备上
    #         images = images.to(device)
    #         labels = labels.to(device)
    #
    #         # 前向传播
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         # 反向传播
    #         optomizer.zero_grad()
    #         loss.backward()
    #         optomizer.step()
    #         total_loss += loss
    #         print(f"Epoch [{epoch + 1}/{epoches}], Iter [{i}/{len(train_dataloader)}], Loss {loss:.4f}")
    #     avg_loss = total_loss / len(train_dataloader)
    #     print(f"Epoch [{epoch + 1}/{epoches}], Loss {avg_loss:.4f}")
    #     if (epoch+1) % 10 == 0:
    #         torch.save(model.state_dict(), f"./model/model_{epoch}.pth")

    print("开始验证/评估模型：")

    model.load_state_dict(
        torch.load(
            r"D:\AI_Learning\python\Deep_Learning\Visual_Classic_Neural_Network\AlexNet_dogs_and_cats\model\best_model.pth",
            map_location=device  # 自动将模型加载到正确的设备上
        )
    )
    # 模型评估
    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    # 测试进度条
    # val_bar = tqdm(test_dataloader, desc=f'Testing')
    with torch.no_grad():
        val_bar = tqdm(test_dataloader, desc=f'Validating')
        for images, labels in val_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            running_val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_bar.set_postfix({'Loss': running_val_loss/1, 'Accuracy': 100 * val_correct / total})

            # 将两者标签移动至CPU并转换为NumPy数组，方便后续计算，绘制混淆矩阵等
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = running_val_loss / len(test_dataloader)
    val_accuracy = 100 * val_correct / total
    print(f'avg_val_Loss: {avg_val_loss:.4f}, Test_Accuracy: {val_accuracy:.4f}%') # 80.50%

    class_names = train_dataset.classes  # 获取数据集的类别名称 ['ants', 'bees']

    display_loader = DataLoader(test_dataset, 
                            batch_size=16,  
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    # 获取一批数据
    display_images, display_labels = next(iter(display_loader))
    # 将图像和标签都移到GPU
    display_images = display_images.to(device)
    display_labels = display_labels.to(device)  # 添加这行，确保标签也在GPU上
    # 获取预测结果
    with torch.no_grad():
        outputs = model(display_images)
        _, display_predicted = torch.max(outputs.data, 1)

    # 显示16张图片，优化布局
    plt.figure(figsize=(15, 15))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        # 将图片从张量转换回numpy，并调整通道顺序
        img = display_images[i].cpu().permute(1, 2, 0)  # 添加.cpu()确保在CPU上处理
        # 反归一化
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = img.numpy()
        # 将像素值裁剪到[0,1]范围内
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # 获取预测标签和真实标签
        pred_label = class_names[display_predicted[i]]
        true_label = class_names[display_labels[i]]
        
        # 设置标签显示在图片下方，两行显示，不堆叠
        color = 'green' if display_predicted[i] == display_labels[i] else 'red'
        
        # 在图片上添加预测和实际标签
        ax = plt.gca()
        # 移除之前的文本标签方式，改用更可靠的方式
        ax.set_xlabel(f'预测: {pred_label}\n实际: {true_label}', color=color, fontsize=9)
        
        # 移除坐标轴刻度，使图片更整洁
        plt.xticks([])
        plt.yticks([])
        
        # 添加边框，使图片之间有明显分隔
        plt.box(on=True)

    # 计算准确率时，确保所有张量都在同一设备上
    accuracy = (display_predicted[:16] == display_labels[:16]).sum().item() / 16 * 100
    plt.suptitle(f'测试图片准确率: {accuracy:.2f}%', fontsize=16)

    # 调整子图之间的间距，使布局更紧凑美观，同时为标签留出足够空间
    plt.subplots_adjust(wspace=0.3, hspace=1.0)  # 大幅增加垂直间距以容纳标签
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为顶部和底部标题留出空间
    plt.show()

    # 生成混淆矩阵
    conf = confusion_matrix(true_labels, predicted_labels) 
    # 可视化
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues") # annot=True 显示数字，fmt="d" 整数格式，cmap="Blues" 颜色
    plt.xlabel("predict labels")
    plt.ylabel("true labels")
    plt.show()




