import numpy as np
import pylab as pl
from sklearn import neighbors, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import time


X_train = np.load("./npy/X_train.npy")
print("load1")
X_test = np.load("./npy/X_test.npy")
print("load2")
Y_train = np.load("./npy/Y_train.npy")
print("load3")
Y_test = np.load("./npy/Y_test.npy")
print("load4")
knn = neighbors.KNeighborsClassifier()
#svm_c = OneVsRestClassifier(svm.SVC(class_weight='balanced'))
c_range = np.array([1, ])
print(c_range)
param_grid = [{'metric': ['manhattan'], 'n_neighbors': c_range}]
model = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
print("ready to fit")
t0 = time.time()
model.fit(X_train,Y_train)
t1 = time.time()
print(t1-t0)
print("fit over")
score = model.score(X_test, Y_test)
print('Score%.3f' % score)
df = pd.DataFrame(model.cv_results_)
cvresults_path = './'
df.to_csv(cvresults_path + 'resultNaiveKNN_score.csv')


print("hello")