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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create FastText embeddings for descriptions')
    parser.add_argument('descriptions', help='the csv file containing the description with good/bad annotations')
    parser.add_argument('--folds', help='folder to split (default = 10)', default='10' ,type=int)

    args = parser.parse_args()

    descriptions = args.descriptions
    print("Processing file {} containing descriptions.".format(descriptions))
    assert os.path.exists(descriptions), "file containing descriptions NON FOUND!!!"

    start = time()

    print("Processing data")

    listLines = []
    Xlist = []
    ylist = []

    with open(descriptions) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            line = "__label__{} {}".format(row[1], row[0])
            listLines.append(line)
            Xlist.append(row[0])
            ylist.append(row[1])
            #print(line)
            line_count += 1
        print(f'Processed {line_count} lines.')

    print("Saving file")

    output = descriptions + ".fbclass.gz"

    f_test = gzip.open(output, "wb")
    np.savetxt(f_test,listLines, fmt="%s", encoding="utf-8")
    f_test.close()

    if "folds" in args:
        cv = args.folds
        #splitting
        skf = StratifiedKFold(n_splits=cv)

        i = 0
        X=np.asarray(Xlist)
        y=np.asarray(ylist)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            i_str = '{:02d}'.format(i)

            listLinesTrain =[]
            for d, l in zip(X_train, y_train):
                listLinesTrain.append("__label__{} {}".format(l, d))

            listLinesTest =[]
            for d, l in zip(X_test, y_test):
                listLinesTest.append("__label__{} {}".format(l, d))

            output_train = descriptions + ".fbclass.train.{}.gz".format(i_str)
            np.savetxt(output_train, listLinesTrain, fmt="%s", encoding="utf-8")

            output_test = descriptions + ".fbclass.test.{}.gz".format(i_str)
            np.savetxt(output_test, listLinesTest, fmt="%s", encoding="utf-8")

            i += 1