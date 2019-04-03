import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'
pca_path = 'Data/pca.npy'
tsne_path = 'Data/tsne3.npy'


def dataset(data_type):
    labels = np.loadtxt(label_path, dtype=int)
    # std_features = 0
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


if __name__ == '__main__':
    make_idx()
