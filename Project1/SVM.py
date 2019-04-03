from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np
import pandas as pd
from Dataset import dataset

feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'
cvresults_path = 'Results/raw-cv'
data_type = 'raw'  # raw, PCA, tsne
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'


if __name__ == '__main__':
    train_fs, test_fs, train_labels, test_labels = dataset(data_type)
    svm_c = svm.SVC(class_weight='balanced')

    c_range = np.logspace(-15, -1, num=7, base=2)
    # gamma_range = np.logspace(-13, -13, num=1, base=2)
    param_grid = [{'kernel': ['linear'], 'C': c_range}]    # , 'gamma': gamma_range
    grid = GridSearchCV(svm_c, param_grid, cv=5, n_jobs=-1)
    grid.fit(train_fs, train_labels)
    score = grid.score(test_fs, test_labels)
    print('Score%.3f' % score)
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(cvresults_path + '%.3f.csv' % score)

