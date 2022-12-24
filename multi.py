import numpy as np
import pandas as pd
import cv2
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


def pre_process(idx: int, typ: int = 1, kernel=[[1, -1]], indexes=[0]):
    # 预处理，返回特征向量
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
    # 提取特征，返回数据集
    data = []
    for i in range(nr_imgs):
        data.append(pre_process(i, 0, kernel, indexes))
        data.append(pre_process(i, 1, kernel, indexes))
    data = np.array(data)
    np.random.seed(seed)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


def train(data_x, data_y, models: list):
    # 批量训练，返回结果
    res = []
    for model in models:
        prediction = cross_val_predict(model, data_x, data_y, cv=10)
        acc = round(accuracy_score(data_y, prediction), 2)
        precision = round(precision_score(data_y, prediction), 2)
        recall = round(recall_score(data_y, prediction), 2)
        f = round(f1_score(data_y, prediction), 2)
        print(f"[{model}]: 精度:{acc}, 查准率:{precision}, 查全率:{recall}, f1值:{f}")
        res += [acc, precision, recall, f]
    return res


def get_indexes_by_bin(b: int, indexes: list):
    # 二进制转索引
    idx_str = reversed(bin(b)[2:])
    res = []
    for (i, c) in enumerate(idx_str):
        if c == '1' and i < len(indexes):
            res.append(indexes[i])
    return res


def run_test(rng):
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    import warnings
    warnings.filterwarnings("ignore")
    models = [GaussianNB(), BernoulliNB(), MultinomialNB()]
    indexes = [0, 1, 2, 3, 4, 5, 8, 16]
    res_list = []
    for i in rng:
        idx = get_indexes_by_bin(i, indexes)
        if len(idx) == 0:
            continue
        print(i, idx)
        data_x, data_y = feature_extract(
            nr_imgs=64, kernel=kernel, indexes=idx)
        res = train(data_x, data_y, models)
        lst = [i, str(idx)] + res
        res_list.append(lst)
    return res_list


if __name__ == '__main__':
    from multiprocessing import Pool

    p = Pool(8) # 创建8个进程
    result = []
    np = []

    total = 256
    for i in range(1, total, total//8): # 每个进程处理total/8个
        np.append(p.apply_async(run_test, args=(range(i, i+total//8),)))

    p.close() # 关闭进程池，不再接受新的进程
    p.join() # 主进程阻塞等待子进程的退出

    for i in np:
        result += i.get()   # 获取子进程的返回值
    list.sort(result, key=lambda x: x[0]) # 排序

    df = pd.DataFrame(result)
    df.to_excel('result.xlsx', index=False) # 保存结果
