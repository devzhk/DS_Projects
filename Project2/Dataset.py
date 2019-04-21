import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import pandas as pd

feature_path = 'Data/features.npy'
label_path = 'Data/AwA2-labels.txt'
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'
vae128_path = 'Data/VAE-128.npy'
vae256_path = 'Data/VAE-256.npy'
vae64_path = 'Data/VAE-64.npy'
vae32_path = 'Data/VAE-32.npy'
vae16_path = 'Data/VAE-16.npy'
b5_vae32_path = 'Data/b5VAE-32.npy'
ae16_path = 'Data/AE-16.npy'
ae32_path = 'Data/AE-32.npy'
ae64_path = 'Data/AE-64.npy'
ae128_path = 'Data/AE-128.npy'
ae256_path = 'Data/AE-256.npy'
b05vae16_path = 'Data/'


def dataset(data_type):
    labels = np.loadtxt(label_path, dtype=int)
    if data_type == 'raw':
        std_features = np.load(feature_path)
        print('Load raw features')
    elif data_type == 'PCA':
        std_features = np.load(pca_path)
        print('Load PCA features')
    elif data_type == 'tsne':
        std_features = np.load(tsne_path)
    # elif data_type == 'VAE-128':
    #     std_features = np.load(vae128_path)
    # elif data_type == 'VAE-256':
    #     std_features = np.load(vae256_path)
    # elif data_type == 'VAE-64':
    #     std_features = np.load(vae64_path)
    # elif data_type == 'VAE-32':
    #     std_features = np.load(vae32_path)
    # elif data_type == 'VAE-16':
    #     std_features = np.load(vae16_path)
    # elif data_type == 'b5VAE-32':
    #     std_features = np.load(b5_vae32_path)
    # elif data_type == 'AE-16':
    #     std_features = np.load(ae16_path)
    # elif data_type == 'AE-32':
    #     std_features = np.load(ae32_path)
    # elif data_type == 'AE-64':
    #     std_features = np.load(ae64_path)
    # elif data_type == 'AE-128':
    #     std_features = np.load(ae128_path)
    # elif data_type == 'AE-256':
    #     std_features = np.load(ae256_path)
    else:
        std_features = np.load('Data/%s.npy' % data_type)
    # else:
    #     print('No such data type')
    print(labels.shape, std_features.shape)
    train_idx = np.load('Data/train_idx.npy')
    test_idx = np.load('Data/test_idx.npy')
    train_f, test_f, train_label, test_label = std_features[train_idx, :], std_features[test_idx, :], \
                                               labels[train_idx], labels[test_idx]
    # print('train stat', pd.value_counts(train_label))
    # print('test stat', pd.value_counts(test_label))
    return train_f, test_f, train_label, test_label


def make_idx():
    indices = np.random.permutation(37322)
    boundary = round(0.6 * 37322)
    train_idx, test_idx = indices[:boundary], indices[boundary:]
    print(train_idx.shape, test_idx.shape)
    np.save('Data/train_idx.npy', train_idx)
    np.save('Data/test_idx.npy', test_idx)


class AwA2(data.Dataset):
    def __init__(self):
        self.features = np.load(feature_path)
        self.labels = np.loadtxt(label_path, dtype=int)

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return self.labels.shape[0]


if __name__ == '__main__':
    # make_idx()
    print('Testing')