import os
import numpy as np
from dataset import get_data,normalize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import cv2

# VAE encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2_mean = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

# VAE decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 32 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 32, 8, 8)
        x = torch.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

# VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# 计算重构损失和KL散度损失
def loss_function(x_recon, x, mean, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return 0.5 * reconstruction_loss + 0.5 * kl_divergence_loss

if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    X_train = torch.from_numpy(X_train).float().to(device)

    # 设置超参数
    latent_dim = 16
    batch_size = 64
    learning_rate = 0.003
    num_epochs = 150

    # 创建数据加载器
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化VAE模型和优化器
    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    Loss = []

    # 训练VAE模型
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            x_recon, mean, logvar = vae(x)
            loss = loss_function(x_recon, x, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(X_train):.4f}")
        Loss.append(total_loss / len(X_train))

    Loss = np.array(Loss)
    plt.plot(Loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.show()

    # 重建
    vae.eval()
    with torch.no_grad():
        imgs_input = X_train[:20]
        reconstructed, mean, logvar = vae(imgs_input)
        reconstructed = torch.Tensor.cpu(reconstructed)
        reconstructed = reconstructed.detach().numpy()
    
    cnt = 0
    for img in reconstructed:
        img = img.transpose(1, 2, 0)[..., ::-1]
        cv2.imwrite('pic{}_re.png'.format(cnt), img * 255)
        cnt += 1

    # 随机生成
    vae.eval()
    # generate z_1 and x_1
    with torch.no_grad():
        z1 = torch.randn(5, latent_dim).to(device)
        samples1 = torch.Tensor.cpu(vae.decoder(z1))
        samples1 = samples1.detach().numpy()

    cnt = 1
    for img in samples1:
        img = img.transpose(1, 2, 0)[..., ::-1]
        cv2.imwrite('interpolation/z_1_{}.png'.format(cnt), img * 255)
        cnt += 1

    # generate z_2 and x_2
    with torch.no_grad():
        z2 = torch.randn(5, latent_dim).to(device)
        samples2 = torch.Tensor.cpu(vae.decoder(z2))
        samples2 = samples2.detach().numpy()

    cnt = 1
    for img in samples2:
        img = img.transpose(1, 2, 0)[..., ::-1]
        cv2.imwrite('interpolation/z_2_{}.png'.format(cnt), img * 255)
        cnt += 1

    # generate z_3 and x_3
    with torch.no_grad():
        z3 = 0.5 * z1 + 0.5 * z2
        samples3 = torch.Tensor.cpu(vae.decoder(z3))
        samples3 = samples3.detach().numpy()

    cnt = 1
    for img in samples3:
        img = img.transpose(1, 2, 0)[..., ::-1]
        cv2.imwrite('interpolation/z_3_{}.png'.format(cnt), img * 255)
        cnt += 1
    
    # 线性插值
    vae.eval()
    with torch.no_grad():
        imgs_input = X_train[:5]
        mean, logvar = vae.encoder(imgs_input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z1 = mean + eps * std

        imgs_input = X_train[10:15]
        mean, logvar = vae.encoder(imgs_input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z2 = mean + eps * std

        for k in range(9):
            z3 = 0.125 * k * z1 + (1 - 0.125 * k) * z2
            samples = torch.Tensor.cpu(vae.decoder(z3))
            samples = samples.detach().numpy()

            cnt = 1
            for img in samples:
                img = img.transpose(1, 2, 0)[..., ::-1]
                cv2.imwrite('ratios/group{}_{}.png'.format(cnt, k), img * 255)
                cnt += 1