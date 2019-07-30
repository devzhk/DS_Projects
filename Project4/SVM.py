from sklearn.svm import SVC
import sklearn.neighbors
import pandas as pd
import numpy as np
import time


def svmmodel(Xs, Ys, Xt, Yt):
    clf = SVC(C=1, gamma='scale', decision_function_shape='ovo')
    print('fitting...')
    clf.fit(Xs, Ys)
    print('predicting...')
    acc = clf.score(Xt, Yt)
    return acc


def knnmodel(Xs, Ys, Xt, Yt):
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    print('fitting...')
    clf.fit(Xs, Ys)
    print('predicting...')
    acc = clf.score(Xt, Yt)
    return acc


if __name__ == '__main__':
    log_path = 'svm_results.txt'
    f = open(log_path, 'a')
    src_paths = ['Office-Home_resnet50/Art_Art.csv', 'Office-Home_resnet50/Clipart_Clipart.csv', 'Office-Home_resnet50/Product_Product.csv' ]
    tar_paths = ['Office-Home_resnet50/Art_RealWorld.csv', 'Office-Home_resnet50/Clipart_RealWorld.csv', 'Office-Home_resnet50/Product_RealWorld.csv']
    for domain_chosen in [0, 1, 2]:
        print(domain_chosen)
        f.write(tar_paths[domain_chosen] + '\n')
        col = np.arange(0, 2048)
        src_path, tar_path = src_paths[domain_chosen], tar_paths[domain_chosen]
        Xs, Ys, Xt, Yt = pd.read_csv(src_path, usecols=col), pd.read_csv(src_path, usecols=[2048]), pd.read_csv(
            tar_path, usecols=col), pd.read_csv(tar_path, usecols=[2048])
        Xs, Ys, Xt, Yt = Xs.values, Ys.values.ravel(), Xt.values, Yt.values.ravel()
        print(Xs.shape, Ys.shape, Xt.shape, Yt.shape)
        t0 = time.time()
        acc = knnmodel(Xs, Ys, Xt, Yt)
        t1 = time.time()
        print('SVM running time: %s seconds\n' % str(t1 - t0))
        print(acc)
        f.write('SVM running time: %s seconds\n' % str(t1 - t0) + \
                'SVM score: %.4f' % acc)