import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

pca_path = 'Data/pca02.npy'
feature_path = 'RawData/AwA2-features.txt'
threshold = 0.9

def pca(path):
    features = np.loadtxt(path, delimiter=' ')
    # scaler = StandardScaler()
    # std_features = scaler.fit_transform(features)
    p = PCA(n_components=threshold)
    # p.fit(std_features)
    p_data = p.fit_transform(features)
    print(features.shape, p_data.shape)
    np.save(pca_path, p_data)

    print(p.explained_variance_ratio_)
    print(p.explained_variance_)


def k_pca(path):
    features = np.loadtxt(path, delimiter=' ')
    p = KernelPCA(n_components=467, kernel='linear')
    p_data = p.fit_transform(features)
    print(features.shape, p_data.shape)
    print(p.explained_variance_ratio_)
    print(p.explained_variance_)


if __name__ == '__main__':
    # pca(feature_path)
    # a = np.load('Data/pca.npy')
    # print(a.shape)
    k_pca(feature_path)