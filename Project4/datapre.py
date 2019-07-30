import pandas as pd 
import numpy as np 
import scipy.io as io

def convert(src_paths):
    print('Converting: ', src_paths)
    for path in src_paths:
        Xs, Ys = pd.read_csv(path, usecols=np.arange(0,2048)), pd.read_csv(path, usecols=[2048])
        Xs, Ys = Xs.values, Ys.values.ravel()
        print(Xs.shape, Ys.shape)
        io.savemat(path.replace('.csv', '.mat'), {'fts':Xs, 'labels':Ys})


if __name__ == "__main__":
    src_paths = ['Home/Art_Art.csv', 'Home/Clipart_Clipart.csv', 'Home/Product_Product.csv' ]
    tar_paths = ['Home/Art_RealWorld.csv', 'Home/Clipart_RealWorld.csv', 'Home/Product_RealWorld.csv']
    convert(src_paths)
    convert(tar_paths)