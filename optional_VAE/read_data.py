import os
import numpy as np
from dataset import get_data,normalize

from matplotlib import pyplot as plt

if __name__ == "__main__":
    X_train = get_data('dataset')

    for k in range(20):
        img = np.array([[[X_train[k][0][i][j], X_train[k][1][i][j], X_train[k][2][i][j]] for j in range(32)] for i in range(32)])
        print(img.shape)
        plt.imsave("pic{}.png".format(k), img)
