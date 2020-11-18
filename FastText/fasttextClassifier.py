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
    parser.add_argument('base', help='base path for the train / test files')

    args = parser.parse_args()

    descriptions = args.base
    print("Processing file {} containing descriptions.".format(descriptions))

    start = time()

    print("Processing data")

    final_pred = list()
    final_test = list()

    for i in range(10):
        i_str = '{:02d}'.format(i)

        train_file = descriptions + ".train." + i_str
        test_file = descriptions + ".test." + i_str

        print("Cross-validating on iteration {}".format(i_str))

        start = time()

    #Default embeddings

        model = fasttext.train_supervised(input=train_file,
                                   lr=1.0,
                                   epoch=100,
                                   wordNgrams=2,
                                   bucket=200000,
                                   dim=50,
                                   loss='hs')

'''      
    #Pre-trained embeddings wikipedia uncomment to use

        model = fasttext.train_supervised(
                                input=train_file,
                                lr=1.0, epoch=100,
                                wordNgrams=2, bucket=200000, dim=300, loss='hs',
                                pretrainedVectors='cc_300.vec')

'''    
        print("Training time: {}".format(time()-start))
        start = time()
        with open(test_file, 'r') as f:
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
            elif line.strip():
                print("<EMPTY?")
                print(line)
                print(">")
            else:
                print("<ERROR reading test")
                print(line)
                print(">")

            predLabel = model.predict(desc.rstrip("\n\r"))[0][0];
            if predLabel == "__label__Good":
                pred = 1
            elif predLabel == "__label__Bad":
                pred = 0
            else:
                print("ERROR in prediction")

            listPred.append(pred)
            listLabel.append(label)

        print("Testing time: {}".format(time() - start))

        final_pred.extend(listPred)
        final_test.extend(listLabel)

    print(final_test)
    print(final_pred)

    output = descriptions + ".eval.gz"
    with gzip.open(output, "wb") as f:
        np.savetxt(f, (final_test, final_pred), fmt='%i')