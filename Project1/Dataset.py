import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data
feature_path = 'RawData/features.npy'
label_path = 'RawData/AwA2-labels.txt'
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'
vae128_path = 'Data/VAE-128.npy'

def dataset(data_type):
    labels = np.loadtxt(label_path, dtype=int)
    if data_type == 'raw':
        features = np.loadtxt(feature_path, delimiter=' ')
        scaler = StandardScaler()
        std_features = scaler.fit_transform(features)
        print('Load raw features')
    elif data_type == 'PCA':
        std_features = np.load(pca_path)
        print('Load PCA features')
    elif data_type == 'tsne':
        std_features = np.load(tsne_path)
    elif data_type == 'VAE-128':
        std_features = np.load(vae128_path)
    else:
        print('No such data type')
    print(labels.shape, std_features.shape)
    train_idx = np.load('Data/train_idx.npy')
    test_idx = np.load('Data/test_idx.npy')
    train_f, test_f, train_label, test_label = std_features[train_idx, :], std_features[test_idx, :], \
                                               labels[train_idx], labels[test_idx]
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
        print(self.features.shape)
        self.labels = np.loadtxt(label_path, dtype=int)

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return self.labels.shape[0]

if __name__ == '__main__':
    make_idx()
