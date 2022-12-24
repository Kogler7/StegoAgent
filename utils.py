import numpy as np
import pandas as pd
import cv2
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

kernels = [
    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
    [-1, 1],
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
]

base_indexes = [0, 1, 2, 3, 4, 5, 8, 16, 32, 64]


def pre_process(idx: int, typ: int = 1, kernel=[[1, -1]], indexes=[0]):
    src = 'Stego' if typ == 1 else 'Cover'
    img = cv2.imread('./data' + f'/{src}/{idx+1}.pgm')
    cov = cv2.filter2D(img, cv2.CV_16S, kernel=np.array(kernel))
    arr = np.array(cov).flatten()
    ans = Counter(arr)
    ans = np.array([(ans[i]+ans[-i]) for i in indexes]).flatten()
    cv2.normalize(ans, ans, 0, 1, cv2.NORM_MINMAX)
    ans = np.append(ans, typ)
    return ans


def feature_extract(seed: int = 0, nr_imgs: int = 100, kernel=[[1, -1]], indexes: list = [0]):
    data = []
    for i in range(nr_imgs):
        data.append(pre_process(i, 0, kernel, indexes))
        data.append(pre_process(i, 1, kernel, indexes))
    data = np.array(data)
    np.random.seed(seed)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


def train(data_x, data_y, models: list, cross=cross_val_predict):
    res = []
    for model in models:
        prediction = cross(model, data_x, data_y, cv=10)
        acc = round(accuracy_score(data_y, prediction), 2)
        precision = round(precision_score(data_y, prediction), 2)
        recall = round(recall_score(data_y, prediction), 2)
        f = round(f1_score(data_y, prediction), 2)
        print(f"[{model}]: 精度:{acc}, 查准率:{precision}, 查全率:{recall}, f1值:{f}")
        res += [acc, precision, recall, f]
    return res


def get_indexes_by_bin(b: int, indexes: list):
    idx_str = reversed(bin(b)[2:])
    res = []
    for (i, c) in enumerate(idx_str):
        if c == '1' and i < len(indexes):
            res.append(indexes[i])
    return res
