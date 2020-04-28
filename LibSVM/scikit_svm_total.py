import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from time import time

#OAV Training set

r_OAV60000_trainingTSV = 'TSV/vect_300dim/OAV_60000/OAV_60000_Training48000.tsv'

OAV60000_training_read = pd.read_csv(r_OAV60000_trainingTSV, sep='\t',names=["vector"])

df1 = pd.DataFrame(OAV60000_training_read)

df1 = pd.DataFrame(df1.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df1)

#RA Training set

r_RA_30251__trainingTSV = 'TSV/vect_300dim/RA_30251/RA_30251_Training24201.tsv'

RA_30251__training_read = pd.read_csv(r_RA_30251__trainingTSV, sep='\t',names=["vector"])

df2 = pd.DataFrame(RA_30251__training_read)

df2 = pd.DataFrame(df2.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df2)


#A Training set

r_A19784_trainingTSV = 'TSV/vect_300dim/A19859/A19784_Training15898.tsv'

A19859_training_read = pd.read_csv(r_A19784_trainingTSV, sep='\t',names=["vector"])

df3 = pd.DataFrame(A19859_training_read)

df3 = pd.DataFrame(df3.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df3)

#OAV Test set

r_OAV60000_testTSV = 'TSV/vect_300dim/OAV_60000/OAV_60000_Test12000.tsv'

OAV60000_test_read = pd.read_csv(r_OAV60000_testTSV, sep='\t',names=["vector"])

df4 = pd.DataFrame(OAV60000_test_read)

df4 = pd.DataFrame(df4.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df4)

#RA Test set

r_RA_30251__testTSV = 'TSV/vect_300dim/RA_30251/RA_30251_Test6050.tsv'

RA_30251__test_read = pd.read_csv(r_RA_30251__testTSV, sep='\t',names=["vector"])

df5 = pd.DataFrame(RA_30251__test_read)

df5 = pd.DataFrame(df5.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df5)


#A Test set

r_A19784_testTSV = 'TSV/vect_300dim/A19859/A19784_test3886.tsv'

A19859_test_read = pd.read_csv(r_A19784_testTSV, sep='\t',names=["vector"])

df6 = pd.DataFrame(A19859_test_read)

df6 = pd.DataFrame(df6.vector.str.split(' ',1).tolist(),
                                   columns = ['label','vector'])

print(df6)

#Concatenate Training set


training_set = pd.concat([df2, df3], ignore_index=True)

#Concatenate Test set

#test_set = pd.concat([df4, df5, df6],  ignore_index=True) 

print(training_set.label)

y_Training = pd.DataFrame([training_set.label]).astype(int).to_numpy().reshape(-1,1).ravel()
print(y_Training.shape)


X_Training = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in training_set['vector']])
print(X_Training.astype(float).to_numpy())
print(X_Training)


y_Test = pd.DataFrame([df4.label]).astype(int).to_numpy().reshape(-1,1).ravel()
print(y_Test.shape)


X_Test = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df4['vector']])
print(X_Test.astype(float).to_numpy())
print(X_Test)



start = time()
X_train, y_train = 	X_Training, y_Training
X_test, y_test = 	X_Test, y_Test


clf = svm.SVC(kernel='rbf',
              C=3,
              gamma=3,
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
print("Training test prediction (80-20)seconds: {}".format(time() - start))
print(confusion_df)