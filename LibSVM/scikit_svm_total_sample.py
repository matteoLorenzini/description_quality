import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from time import time
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


# Generate 20k sample
X_sample, y_sample = resample (X, y, n_samples=100000, replace=False, stratify=y,random_state=42)

print("Start split and prediction")

start = time()

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2,random_state=0)



clf = svm.SVC(kernel='rbf',
              C=1,
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









