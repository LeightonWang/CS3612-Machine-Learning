import os
import numpy as np
from dataset import get_data,normalize

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from torchvision import transforms
from sklearn.manifold import TSNE


def my_pca(X):
    # 样本中心化
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    
    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 选取前两个特征向量对应的特征值最大的两个主成分
    top_two_eigenvectors = eigenvectors[:, :2]
    
    # 将数据投影到选取的主成分上
    transformed_data = np.dot(X_centered, top_two_eigenvectors)
    
    return transformed_data

def pairwise_distances(X):
    sum_squared = np.sum(X ** 2, axis=1)
    distances = -2 * np.dot(X, X.T) + sum_squared + sum_squared[:, np.newaxis]
    distances = np.maximum(distances, 0)  # Ensure distances are non-negative
    return np.sqrt(distances)


def compute_joint_probabilities(distances, perplexity=30.0, epsilon=1e-4):
    N = distances.shape[0]
    P = np.zeros((N, N))
    perplexity = min(perplexity, N - 1)

    for i in range(N):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search to find the appropriate beta value
        for _ in range(50):
            sum_Pi = np.sum(np.exp(-distances[i, :] * beta))
            sum_Pi -= np.exp(-distances[i, i] * beta)
            entropy = np.log(sum_Pi) + beta * np.sum(distances[i, :] * np.exp(-distances[i, :] * beta)) / sum_Pi

            # Update the beta value based on the perplexity
            if entropy < np.log(perplexity):
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        # Compute the final pairwise probabilities
        P[i, :] = np.exp(-distances[i, :] * beta)
        P[i, i] = epsilon

        # Normalize the probabilities
        P[i, :] /= np.sum(P[i, :])

    return P


def compute_gradient(Y, P):
    N = Y.shape[0]
    grads = np.zeros((N, 2))

    for i in range(N):
        diff = Y[i, :] - Y
        grad = 4.0 * ((P[i, :] - P[i, :].dot(P[i, :].T)).dot(diff))
        grads[i, :] = grad

    return grads


def t_sne(X, num_iterations=50, learning_rate=1e-3, perplexity=30.0, num_dims=2, verbose=True):
    """
    t-SNE algorithm implementation.
    :param X: Input data matrix (N x D)
    :param num_iterations: Number of iterations to run
    :param learning_rate: Learning rate for gradient descent
    :param perplexity: Perplexity value
    :param num_dims: Number of dimensions for the low-dimensional representation
    :param verbose: Whether to print progress during iterations
    :return: Low-dimensional representation of X (N x num_dims)
    """
    N, D = X.shape

    # Initialize the low-dimensional representation randomly
    Y = np.random.randn(N, num_dims)

    # Compute pairwise distances
    distances = pairwise_distances(X)

    # Compute joint probabilities
    P = compute_joint_probabilities(distances, perplexity)

    for iteration in range(num_iterations):
        # Compute gradients
        grads = compute_gradient(Y, P)

        # Update the low-dimensional representation
        Y -= learning_rate * grads

        # Zero mean of the low-dimensional representation
        Y -= np.mean(Y, axis=0)

        # Print progress
        # if verbose and iteration % 50 == 0:

        #     cost = np.sum(P * np.log(P / (distances + 1e-3)))
        #     print(f"Iteration {iteration}: Cost={cost}")

    return Y

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
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(100, num_classes)
        
        
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
        # x = self.relu4(x)
        # x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
def get_intermediate_features(model, x):
    intermediate_features = []
    # 注册钩子函数，提取中间层输出特征
    def hook_fn(module, input, output):
        intermediate_features.append(output)
    # 注册钩子函数到中间层
    target_layer = model.relu2  # 设置为你要提取特征的中间层
    hook_handle = target_layer.register_forward_hook(hook_fn)
    # 前向传播
    model(x)
    # 移除钩子函数
    hook_handle.remove()
    # 返回中间层输出特征
    return intermediate_features[0]


if __name__ == "__main__":

    X_train, X_test, Y_train, Y_test = get_data('dataset')
    test_dataset = CustomDataset(X_test, Y_test)
    
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100

    # 加载已训练的网络模型
    model = MyNet()
    model.load_state_dict(torch.load("MyNet.pth"))

    # 加载数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 选择中间某一层的输出作为特征
    layer_name = 'fc2'
    layer_output = []

    # 定义钩子函数以获取中间层的输出
    def get_layer_output_hook(module, input, output):
        layer_output.append(output)

    # 注册钩子函数到指定层
    target_layer = getattr(model, layer_name)
    hook = target_layer.register_forward_hook(get_layer_output_hook)

    # 运行模型并获取特征
    for images, labels in test_loader:
        _ = model(images)

    # 取消钩子函数的注册
    hook.remove()

    # 将特征转换为numpy数组
    features = torch.cat(layer_output).detach().numpy()
    features = features.reshape(features.shape[0], -1)

    # 原始数据
    # features = X_test.reshape(X_test.shape[0], -1)

    print(features.shape)

    # 创建PCA模型并拟合特征数据
    visualizer = TSNE(n_components=2)
    visualized_features = visualizer.fit_transform(features)
    # visualized_features = t_sne(features)
    # visualized_features = my_pca(features)

    # 绘制散点图可视化
    plt.scatter(visualized_features[:, 0], visualized_features[:, 1], c=Y_test, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of {} Layer'.format(layer_name))
    # plt.title('Visualization for Initial data')
    plt.show()
