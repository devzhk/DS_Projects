from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np
import pandas as pd
from Dataset import dataset

feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'

type_list = ['raw', 'PCA', 'tsne', 'VAE-16', 'VAE-32', 'VAE-64', 'VAE-128', 'VAE-256', 'b5VAE-32',
             'AE-16', 'AE-32', 'AE-64', 'AE-128', 'AE-256']
data_type = type_list[-1]  # raw, PCA, tsne，VAE-128
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'
cvresults_path = 'Results/%s-9-0-cv2_' % data_type


def preview(train_label, test_label):
    print('train stat', pd.value_counts(train_label))
    print('test stat', pd.value_counts(test_label))


if __name__ == '__main__':
    train_fs, test_fs, train_labels, test_labels = dataset(data_type)
    print('Load %s' % data_type)
    # preview(train_labels, test_labels)
    svm_c = svm.SVC()  # class_weight='balanced'

    c_range = np.logspace(-9, 0, num=10, base=2)
    # gamma_range = np.logspace(-13, -13, num=1, base=2)
    param_grid = [{'kernel': ['linear'], 'C': c_range}]    # , 'gamma': gamma_range
    grid = GridSearchCV(svm_c, param_grid, cv=5, n_jobs=-1)
    grid.fit(train_fs, train_labels)
    score = grid.score(test_fs, test_labels)
    print('Score%.3f' % score)
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(cvresults_path + '%.3f.csv' % score)

