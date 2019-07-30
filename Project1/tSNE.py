from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'
tsne_path = 'Data/tsne3.npy'
dim = 3  # < 4


def visual_tsne(std_features, labels):
    plt.figure(figsize=(32, 32))
    for i in range(labels.shape[0]):
        plt.text(std_features[i, 0], std_features[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    labels = np.loadtxt(label_path, dtype=int)
    features = np.loadtxt(feature_path)
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=501)
    tsne_features = tsne.fit_transform(features)
    scaler = StandardScaler()
    std_features = scaler.fit_transform(tsne_features)
    visual_tsne(std_features, labels)
    print(std_features.shape)
    np.save(tsne_path, std_features)
