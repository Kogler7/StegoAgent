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

kernal = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

cover_cov = cv2.filter2D(covers[0], cv2.CV_8U, kernal)
cover_arr = np.array(cover_cov).flatten()
print(cover_arr)
cover_hist = cv2.calcHist([cover_cov], [0], None, [256], [0, 256]) # 256个bin, 0-256

# print(cover_hist)


# print(covers.shape)

# # 获取cover的灰度直方图 256个bin
# cover_hist = cv2.calcHist([covers[0]], [0], None, [256], [0, 256])
# stego_hist = cv2.calcHist([stegos[0]], [0], None, [256], [0, 256])
# 展示输出
plt.plot(cover_hist)
# plt.plot(stego_hist)
plt.legend(['cover', 'stego'])
plt.show()

# diff = covers - stegos

# print(covers)
# print('--------------===========--------------')
# print(stegos)
# print('--------------===========--------------')
# print(diff)
# plt.imshow(diff.squeeze(), cmap='gray')
# plt.show()

# for i in range(100):
#     cover = cv2.imread(dataset_path + f'\\Cover\\{i+1}.pgm')
#     stego = cv2.imread(dataset_path + f'\\Stego\\{i+1}.pgm')
#     plt.clf()
#     cover_hist = cv2.calcHist([cover], [0], None, [256], [0, 256])
#     stego_hist = cv2.calcHist([stego], [0], None, [256], [0, 256])
#     # 展示输出
#     plt.plot(cover_hist)
#     plt.plot(stego_hist)
#     plt.legend(['cover', 'stego'])
#     plt.savefig(dataset_path + f'\\hists\\{i+1}.png')

# print(covers)

# %%
