print(__doc__) 

import pandas as pd
import numpy as np
import pylab as pl

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection._split import train_test_split

r_filenameTSV = 'TSV/A19784_test3886.tsv'

tsv_read = pd.read_csv(r_filenameTSV, sep='\t',names=["vector"])

df = pd.DataFrame(tsv_read)

df = pd.DataFrame(df.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df)


y = pd.DataFrame([df.label]).astype(int).to_numpy().reshape(-1,1).ravel()
print(y.shape)
#exit()

X = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df['vector']])
print(X.astype(float).to_numpy())
print(X)
#exit()

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the trainingset and
# just applying it on the test set.

scaler = StandardScaler()

X = scaler.fit_transform(X)

# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

#C_range = 10. ** np.arange(-3, 8)
#gamma_range = 10. ** np.arange(-5, 4)

param_grid = {
"kernel": ["rbf"],
        "gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        "C": [1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
 }

#param_grid = dict(gamma=gamma_range, C=C_range)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=10)

grid.fit(X, y)

print("The best classifier is: ", grid.best_estimator_)
