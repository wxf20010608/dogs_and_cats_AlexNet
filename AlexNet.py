import torch
from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( # 特征提取层
            nn.Conv2d(3, 96, 11, 4, 2), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.MaxPool2d(3, 2), # kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.MaxPool2d(3, 2), # kernel_size, stride
            nn.Conv2d(256, 384, 3, 1, 1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.Conv2d(384, 384, 3, 1, 1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.Conv2d(384, 256, 3, 1, 1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.MaxPool2d(3, 2), # kernel_size, stride
        )
        self.classifier = nn.Sequential( # 分类层
            nn.Dropout(p=0.5), # p=0.5表示随机丢弃50%的神经元
            nn.Linear(256 * 6 * 6, 4096), # in_features, out_features
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量
            nn.Dropout(p=0.5), # p=0.5表示随机丢弃50%的神经元
            nn.Linear(4096, 4096), # in_features, out_features
            nn.ReLU(inplace=True), # inplace=True表示直接在原变量上修改，而不是返回一个新的变量 
            nn.Linear(4096, num_classes), # in_features, out_features
        )

    def forward(self, x):
        x = self.features(x) # 特征提取层
        x = torch.flatten(x, 1) # 展平
        x = self.classifier(x) # 分类层
        return x


if __name__ == '__main__':
    model = AlexNet() # .to("cuda")
    print(summary(model, (3, 224, 224))) # 输入图片的大小
























