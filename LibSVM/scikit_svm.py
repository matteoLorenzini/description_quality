import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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
              C=100,
              gamma=0.001,
              )

#Train the model using the training sets
clf.fit (X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print ("Metrics and Scoring:")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))

print ("Classification Report:")
print (metrics.classification_report(y_test, y_pred,labels=[0,1]))

print ("Confusion Matrix:")

confusion_df = pd.DataFrame(confusion_matrix(y_test,y_pred),
             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
             index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df)