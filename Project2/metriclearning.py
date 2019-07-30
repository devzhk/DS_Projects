import numpy as np
from metric_learn.lmnn import LMNN
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from Dataset import dataset
from functools import wraps
import time
feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'

type_list = ['raw', 'PCA', 'tsne', 'VAE-16', 'VAE-32', 'VAE-64', 'VAE-128', 'VAE-256', 'b5VAE-32',
             'AE-16', 'AE-32', 'AE-64', 'AE-128', 'AE-256',
             'b0.5VAE-16', 'b0.5VAE-32', 'b0.5VAE-64', 'b0.5VAE-128', 'b0.5VAE-256',
             'b0.1VAE-16', 'b0.1VAE-32', 'b0.1VAE-64', 'b0.1VAE-128', 'b0.1VAE-256']

data_type = type_list[1]  # raw, PCA, tsne，VAE-128
cvresults_path = 'Results/%s-ml' % data_type




def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        results = function(*args, **kwargs)
        t1 = time.time()
        print('Total running time: %s minutes' % str((t1 - t0) / 60))
        return results
    return function_timer


def preview(train_label, test_label):
    print('train stat', pd.value_counts(train_label))
    print('test stat', pd.value_counts(test_label))


def lmnn_knn(k_neighbors, train_fs, test_fs, train_labels, test_labels):
    log_path = 'Results/lmnn-%dnn.txt' % k_neighbors
    m_path = 'Data/lmnnM-%dnn-%s.npy' % (k_neighbors, data_type)
    f = open(log_path, 'a')

    print('lmnn:')
    t0 = time.time()
    lmnn = LMNN(k=k_neighbors, learn_rate=1e-6)
    train_fs = lmnn.fit_transform(train_fs, train_labels)
    M = lmnn.metric()
    np.save(m_path, M)
    test_fs = lmnn.transform(test_fs)
    t1 = time.time()
    f.write('lmnn M matrix saved at %s\n' % m_path)
    print('lmnn running time: %s seconds' % str(t1 - t0))
    f.write('lmnn running time: %s seconds \n' % str(t1 - t0))
    print('knn:')
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, p=2, n_jobs=-1)
    knn.fit(train_fs, train_labels)
    score = knn.score(test_fs, test_labels)
    t1 = time.time()
    print('knn running time: %s seconds' % str(t1 - t0))
    print('knn score: %.4f' % score)
    f.write('knn running time: %s seconds\n' % str(t1 - t0) + \
            'knn score: %.4f' % score)


if __name__ == '__main__':
    train_fs, test_fs, train_labels, test_labels = dataset(data_type)

    for k_neighbors in [2, 3, 4, 5]:
        lmnn_knn(k_neighbors, train_fs, test_fs, train_labels, test_labels)




