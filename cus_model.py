import numpy as np
import random
import torch
from sklearn.model_selection import KFold


class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_data = None
        self.y_data = None

    def fit(self, X, y):
        self.X_data = X
        self.y_data = y

    def predict(self, X):
        y_predict = [self.classify(i) for i in X]
        return np.array(y_predict)

    def classify(self, x):
        New_x = np.tile(x, (self.X_data.shape[0], 1))
        dist = (np.sum((New_x - self.X_data) ** 2, axis=1)) ** 0.5
        nearest = np.argsort(dist)
        topK_y = [self.y_data[i] for i in nearest[:self.k]]
        count = [0 for i in range(self.k)]
        for i in range(self.k):
            count[int(topK_y[i])] += 1
        return np.argmax(count)


class CustomLogistic:
    def __init__(self, batch_size=32, num_epochs=100, lr=0.01):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0

    @staticmethod
    def data_iter(batch_size, features, labels):
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples - batch_size, batch_size):
            j = torch.LongTensor(
                indices[i: min(i + batch_size, num_examples)])
            yield features.index_select(0, j), labels.index_select(0, j)

    def sgd(self, params, lr):
        for param in params:
            param.data -= lr * param.grad

    def net(self, x):
        y_1 = torch.matmul(x, self.w1.T) + self.b1
        y_2 = torch.matmul(y_1, self.w2.T) + self.b2
        y_2 = torch.sigmoid(y_2)
        return y_2

    def fit(self, X, y):
        loss = torch.nn.BCELoss()
        num_input, num_output, num_hidden = len(X[0]), 1, 64
        self.w1 = torch.tensor(np.random.normal(
            0, 0.01, (num_hidden, num_input)), dtype=torch.float32)
        self.b1 = torch.zeros(
            (self.batch_size, num_hidden), dtype=torch.float32)
        self.w2 = torch.tensor(np.random.normal(
            0, 0.01, (num_output, num_hidden)), dtype=torch.float32)
        self.b2 = torch.zeros(
            (self.batch_size, num_output), dtype=torch.float32)
        params = [self.w1, self.b1, self.w2, self.b2]
        for param in params:
            param.requires_grad_(requires_grad=True)
        for epoch in range(self.num_epochs):
            train_l = 0
            for x, y in CustomLogistic.data_iter(self.batch_size, X, y):
                y_hat = self.net(x)
                l = loss(y_hat, y)
                l.backward()
                self.sgd(params, self.lr)
                train_l += l.item()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_hat = self.net(X)
        y_hat = torch.where(y_hat > 0.5, torch.ones_like(
            y_hat), torch.zeros_like(y_hat))
        return y_hat


class CustomSVM:
    def __init__(self, epochs: int = 100, iters: int = 10, lm: float = 0.0001, lr: float = 0.01):
        self.epochs = epochs
        self.iters = iters
        self.lm = lm
        self.lr = lr
        self.W = None

    def fit(self, X, y):
        nr_samples = X.shape[0]
        nr_features = X.shape[1]
        self.W = np.zeros(nr_features, dtype=np.float32)
        sample = np.zeros(nr_features, dtype=np.float32)
        grad = np.zeros(nr_features, dtype=np.float32)
        for i in range(self.iters):
            # 每次迭代随机选择一个训练样本
            index = np.random.randint(0, nr_samples)
            sample = X[index]

            # 计算梯度
            WX = np.dot(self.W, sample)
            if 1 - WX * y[index] > 0:
                grad = self.lm * self.W - sample * y[index]
            else:  # 1-WX *y <= 0的时候，目标函数的前半部分恒等于0, 梯度也是0
                grad = self.lm * self.W

            # 更新权重, lr是学习速率
            self.W = self.W - self.lr * grad

    def predict(self, X):
        if self.W is None:
            return None
        nr_samples = X.shape[0]
        res = np.ones(nr_samples, dtype=np.float32)*(-1)
        for i in range(nr_samples):
            sample = X[i]
            sum = np.dot(self.W, sample)
            if sum > 0:  # 权值>0，认为目标值为1
                res[i] = 1
        return res


def custom_cross_val_predict(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True)
    y_pred = np.zeros(y.shape[0], dtype=np.float32)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
    return y_pred


class CustomLDA(object):
    def __init__(self):
        self.mu_i = 0
        self.mu = 0
        self.cov_i = []
        self.cov = 0
        self.X = 0
        self.y = 0
        self.classes = 0
        self.priorProbs = 0
        self.n_samples = 0
        self.n_features = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape
        self.classes, y_idx = np.unique(y, return_inverse=True)
        self.priorProbs = np.bincount(y) / self.n_samples
        means = np.zeros((len(self.classes), self.n_features))
        np.add.at(means, y_idx, X)
        self.mu_i = means / np.expand_dims(np.bincount(y), 1)  # 每个类别的均值
        self.mu = np.dot(np.expand_dims(
            self.priorProbs, axis=0), self.mu_i)  # 整体均值
        self.cov_i = [np.cov(X[y == group].T)
                      for idx, group in enumerate(self.classes)]  # 每个类别的协方差
        self.cov = sum(self.cov_i) / len(self.cov_i)  # 整体协方差矩阵

    def predict_probs(self, X):
        X = np.array(X)
        Sigma = self.cov
        U, S, V = np.linalg.svd(Sigma)
        Sn = np.linalg.inv(np.diag(S))
        Sigma_inv = np.dot(np.dot(V.T, Sn), U.T)
        # 线性判别函数值, 求出分类概率
        value = np.dot(np.dot(X, Sigma_inv), self.mu_i.T) - \
            0.5 * np.multiply(np.dot(self.mu_i, Sigma_inv).T, self.mu_i.T).sum(axis=0).reshape(1, -1) + \
            np.log(np.expand_dims(self.priorProbs, axis=0))
        likelihood = np.exp(value - value.max(axis=1)[:, np.newaxis])
        pred_probs = likelihood / likelihood.sum(axis=1)[:, np.newaxis]
        return pred_probs

    def predict(self, X):
        pred_probs = self.predict_probs(X)
        pred_value = np.argmax(pred_probs, axis=1)
        return pred_value
