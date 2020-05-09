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
from sklearn.utils import resample
from sklearn.utils import shuffle

#OAV dataset

r_OAV60000_trainingTSV = 'TSV/vect_300dim/OAV_60000/OAV_60000.tsv'

OAV60000_training_read = pd.read_csv(r_OAV60000_trainingTSV, sep='\t',names=["vector"])

df1 = pd.DataFrame(OAV60000_training_read)

df1 = pd.DataFrame(df1.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print("OAV processed")

#RA dataset

r_RA_30251__trainingTSV = 'TSV/vect_300dim/RA_30251/RA_30251.tsv'

RA_30251__training_read = pd.read_csv(r_RA_30251__trainingTSV, sep='\t',names=["vector"])

df2 = pd.DataFrame(RA_30251__training_read)

df2 = pd.DataFrame(df2.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print("RA processed")


#A Dataset

r_A19784_trainingTSV = 'TSV/vect_300dim/A19859/A19784.tsv'

A19859_training_read = pd.read_csv(r_A19784_trainingTSV, sep='\t',names=["vector"])

df3 = pd.DataFrame(A19859_training_read)

df3 = pd.DataFrame(df3.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])


print("A processed")

#Concatenate Training set


df_total = pd.concat([df1,df2, df3], ignore_index=True)



y = pd.DataFrame([df_total.label]).astype(int).to_numpy().reshape(-1, 1).ravel()
print(y.shape)

X = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df_total['vector']])
print(X.astype(float).to_numpy())

print("Start 10 KFold validation")


# Generate 20k sample
X, y = resample (X, y, n_samples=100000, replace=False, stratify=y,random_state=42)


start = time()

clf = svm.SVC(kernel='rbf',
              C=1 ,
              gamma=3,
              )


originalclass = []
predictedclass = []


def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)  # return accuracy score


outer_cv = KFold ( n_splits=10)

# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, 
                               scoring=make_scorer(classification_report_with_accuracy_score))

# Average values in classification report for all folds in a K-fold Cross-validation  
print(classification_report(originalclass, predictedclass))
print("10 folds processing seconds: {}".format(time() - start))









