# import torch
# from torch import nn
# from torch import optim
# from torch.utils.data import Dataset, DataLoader

import numpy as np

import cv2

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

dataset_path = 'C:\\Users\\Kolger\\Desktop\\BIGHOMEWORKONE\\Cover-Stego Image Classification'

covers = np.array([cv2.imread(dataset_path + f'\\Cover\\{i+1}.pgm') for i in range(1)])
stegos = np.array([cv2.imread(dataset_path + f'\\Stego\\{i+1}.pgm') for i in range(1)])

diff = covers - stegos

print(covers)
print('--------------===========--------------')
print(stegos)
print('--------------===========--------------')
print(diff)
# plt.imshow(diff.squeeze(), cmap='gray')
# plt.show()

# for i in range(10000):
#     cover = cv2.imread(dataset_path + f'\\Cover\\{i+1}.pgm')
#     stego = cv2.imread(dataset_path + f'\\Stego\\{i+1}.pgm')
#     diff = cover - stego
#     plt.imshow(diff.squeeze(), cmap='gray')
#     plt.savefig(dataset_path + f'\\diffs\\{i+1}.png')

# print(covers)
