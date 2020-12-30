import time
import numpy as np
import fasttext
import pandas as pd
import csv
import pprint
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import numpy as np
import codecs
from time import time
from nltk.corpus import stopwords
from nltk import download
import csv
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import re
import requests
import sys
import argparse
import gzip
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV

from sklearn.metrics import classification_report

#dataset train
train_set = pd.read_csv ('total/train.csv',sep=',',names=['descriptions', 'label_','domain'])

df_train = pd.DataFrame(train_set)

df_train.loc[df_train['label_'] == 'High_Quality', 'label'] = '__label__Good'  
df_train.loc[df_train['label_'] != 'High_Quality', 'label'] = '__label__Bad'

dfTrain = df_train[['label','descriptions' ]]

print(dfTrain)


#dataset test
test_set = pd.read_csv ('total/test.csv',sep=',',names=['descriptions', 'label_','domain'])

df_test = pd.DataFrame(test_set)

df_test.loc[df_test['label_'] == 'High_Quality', 'label'] = '__label__Good'  
df_test.loc[df_test['label_'] != 'High_Quality', 'label'] = '__label__Bad'

dfTest = df_test[['label','descriptions' ]]

print(dfTest)


dfTest.to_csv('test_label.csv', index = False, header=False, sep=" ", quoting=csv.QUOTE_NONE, escapechar=" ")
dfTrain.to_csv('train_label.csv', index = False, header=False, sep=" ", quoting=csv.QUOTE_NONE, escapechar=" ")


train_file = 'train_label.csv'

test_file = 'test_label.csv' 


final_pred = list()
final_test = list()


print("training model...")
  
    
model = fasttext.train_supervised(input=train_file,
                                lr=1.0, epoch=100,
                                wordNgrams=2, 
                                bucket=200000, 
                                dim=50, 
                                loss='hs')



def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    

result = model.test(test_file)
print_results(*result)

with open(test_file, 'r',encoding="utf8") as f:
    test_desc = f.readlines()

listPred = []
listLabel = []

for line in test_desc:
    if line.startswith("__label__Good "):
        desc = line[len("__label__Good "):]
        label = 1
    elif line.startswith("__label__Bad "):
        desc = line[len("__label__Bad "):]
        label = 0
  
    predLabel = model.predict(desc.rstrip("\n\r"))[0][0];
    if predLabel == "__label__Good":
        pred = 1
    elif predLabel == "__label__Bad":
        pred = 0
    else:
        print("ERROR in prediction")

    listPred.append(pred)
    listLabel.append(label)

print("Testing time: {}".format(time()))

final_pred.extend(listPred)
final_test.extend(listLabel)

print(final_test)
print(final_pred)

print(classification_report(final_pred, final_test, digits=3))