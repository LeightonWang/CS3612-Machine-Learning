import os
import numpy as np
from dataset import get_data,normalize

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(8, 18, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(18 * 5 * 5, 240)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(240, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    ########################################################################
    # 以上加载的数据为 numpy array格式
    # 如果希望使用pytorch或tensorflow等库，需要使用相应的函数将numpy arrray转化为tensor格式
    # 以pytorch为例：
    #   使用torch.from_numpy()函数可以将numpy array转化为pytorch tensor
    #
    # Hint:可以考虑使用torch.utils.data中的class来简化划分mini-batch的操作，比如以下class：
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    ########################################################################

    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Devide: {}".format(device))

    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 100

    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MyNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)
    
    train_loss = []
    test_loss = []

    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播、反向传播和优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练准确度
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item()

        train_loss.append(running_loss)
        train_accuracy.append(100 * correct_train / total_train)

        # 在测试集上评估准确度
        correct_test = 0
        total_test = 0

        running_loss_test = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                tLoss = criterion(outputs, labels)
                running_loss_test += tLoss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # tLoss /= len(test_loader)
        test_accuracy.append(100 * correct_test / total_test)
        test_loss.append(running_loss_test)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.3f} - Train Acc: {train_accuracy[-1]:.2f}% - Test Acc: {test_accuracy[-1]:.2f}%")

    # 绘制损失/准确度变化图表
    print(type(test_loss))
    plt.plot(range(1, num_epochs+1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.show()