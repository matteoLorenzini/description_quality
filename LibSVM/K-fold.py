import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection._validation import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold



r_filenameTSV = 'TSV/A19784.tsv'

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


clf = svm.SVC(kernel='rbf',
              C=3,
              gamma=3,
              )
#scores = cross_val_score(clf, X, y, cv=10)

print ("K-Folds scores:")
#print (scores) 

'''
def classification_report_with_accuracy_score(y_true, y_pred):

    print (classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

scores = cross_val_score(clf, X, y, cv=10, \
    scoring=make_scorer(classification_report_with_accuracy_score))
print (scores)
'''
originalclass = []
predictedclass = []

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score

inner_cv = StratifiedKFold(n_splits=10)
outer_cv = StratifiedKFold(n_splits=10)

# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=make_scorer(classification_report_with_accuracy_score))

# Average values in classification report for all folds in a K-fold Cross-validation  
print(classification_report(originalclass, predictedclass)) 
