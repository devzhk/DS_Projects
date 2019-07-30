from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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
data_type = type_list[0]  # raw, PCA, tsne，VAE-128
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'
cvresults_path = 'Results/eu%s-17-cv_' % data_type


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


if __name__ == '__main__':
    train_fs, test_fs, train_labels, test_labels = dataset(data_type)
    print('Load %s' % data_type)
    # preview(train_labels, test_labels)
    knn = KNeighborsClassifier()
    p = np.array([2])
    k_range = np.array([1, 7])
    param_grid = [{'p': p, 'n_neighbors': k_range}]
    grid = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
    t0 = time.time()
    grid.fit(train_fs, train_labels)
    t1 = time.time()
    print('Total running time: %s seconds' % str(t1 - t0))
    score = grid.score(test_fs, test_labels)
    print('Score%.3f' % score)
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(cvresults_path + '%.3f.csv' % score)

