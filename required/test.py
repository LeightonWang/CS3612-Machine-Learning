import numpy as np

def pairwise_distances(X):
    sum_squared = np.sum(X ** 2, axis=1)
    distances = - 2 * np.dot(X, X.T) + sum_squared + sum_squared[:, np.newaxis]
    distances = np.maximum(distances, 0)  # Ensure distances are non-negative
    return np.sqrt(distances)


def compute_joint_probabilities(distances, perplexity=30.0, epsilon=1e-7):
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


def t_sne(X, num_iterations=1000, learning_rate=200.0, perplexity=30.0, num_dims=2, verbose=True):
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
        if verbose and iteration % 50 == 0:
            cost = np.sum(P * np.log(P / (distances + 1e-7)))
            print(f"Iteration {iteration}: Cost={cost}")

    return Y

# 创建数据集（示例数据）
X = np.random.rand(100, 10)  # 100个样本，每个样本10个特征

# 运行t-SNE算法
X_tsne = t_sne(X)

# 可视化降维结果
import matplotlib.pyplot as plt

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
