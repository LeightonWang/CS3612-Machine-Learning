# CS3612-Machine-Learning
Course Project for CS3612: Machine Learning, 2023 Spring, SJTU

2023春季学期机器学习课程个人代码仓库

## Enviorenment Configuration
- Python 3.8.16
- PyTorch 2.0.1
- Numpy 1.24.3

## Tasks
### Mandatory Task: LeNet
- Understand the structure of LeNet and implement it.
- Design a neural network with 3 to 7 layers based on LeNet, and the network we designed should contain convolution layers, pooling layers and activation layers.
- Train the 2 networks with the given dataset, provide the training and the test result.
- Use both PCA and t-SNE algorithm to visualize the features of the mid layers for both of the networks.

### Optional Task: VAE
- You should design and train a VAE on the dataset.
- You should use Linear interpolation to generate images with specific properties.
  - For example, given two output features of the encoder, i.e., $z_1$ and $z_2$, the decoder takes $z_1$ as the input to generate a face image of a woman, and takes $z_2$ as the input to generate a face image of a man. You can use Linear interpolation to obtain a new feature $z= \alpha z_1 + (1 − \alpha)z_2, \alpha \in (0,1)$. Then, the decoder takes $z$ as the input to generate a new face image. You are required to do experiments with different values of $\alpha$, e.g., $\alpha=0.2, 0.4, 0.6, 0.8$, so as to obtain a set of face images.
  - You can conduct experiments on any two categories (e.g., male and female, old and young).
