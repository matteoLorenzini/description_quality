import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection._validation import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn import decomposition

r_filenameTSV = "TSV/A19784.tsv"


#DF 300 dimension start

tsv_read = pd.read_csv(r_filenameTSV, sep='\t', names=["vector"])

df = pd.DataFrame(tsv_read)

df = pd.DataFrame(df.vector.str.split(" ", 1).tolist(), columns=['label', 'vector'])

print(df)

#DF 300 dimension end

'''
#DF 1 dimension start

tsv_read = pd.read_csv(r_filenameTSV,  names=["vector", "label"])

df = pd.DataFrame(tsv_read)

df = pd.DataFrame(df.vector.str.split("\t", 1).tolist(), columns=['label', 'vector'])

#DF 1 dimension end
'''

y = pd.DataFrame([df.label]).astype(int).to_numpy().reshape(-1, 1).ravel()
print(y.shape)

X = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df['vector']])
print(X.astype(float).to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y)

#PCA decomposition 50 dimension
pca = decomposition.PCA(n_components=50)
pca.fit(X)
X = pca.transform(X)

###end decomposition####

start = time()

clf = svm.SVC(kernel='rbf',
              C=32,
              gamma=8,
              )

print("K-Folds scores:")


originalclass = []
predictedclass = []


def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)  # return accuracy score


#inner_cv = StratifiedKFold(n_splits=10)
outer_cv = StratifiedKFold(n_splits=10)


# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,
                               scoring=make_scorer(classification_report_with_accuracy_score))

# Average values in classification report for all folds in a K-fold Cross-validation  
print(classification_report(originalclass, predictedclass))
print("10 folds processing seconds: {}".format(time() - start))