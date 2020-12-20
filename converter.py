import sys
import json
import fasttext
import numpy as np
import codecs
from time import time
from nltk.corpus import stopwords
# from nltk import download
# download('stopwords')
import os
import re
import requests



def CleanStopWords (sentence):
        stop_words = stopwords.words('italian')
        sentenceClean = RemovePunctuation(sentence)
        sentenceSplitted = sentenceClean.split(" ")
        sentence = [w for w in sentenceSplitted if w not in stop_words]
        return(sentence)

def RemovePunctuation(sentence):
        sentenceClean = re.sub(r'[^A-Za-z0-9]',  ' ', sentence)
        sentenceClean = re.sub(r'\s+',  ' ', sentenceClean)
        return(sentenceClean)

def ExtractVectors(sentence, removeStopwords):

        sentence = str(sentence).lower().rstrip()
        if (removeStopwords):
            sentence = CleanStopWords(sentence)
            sentence = ' '.join(sentence)
        sv = model_ft.get_sentence_vector(sentence)
        vector = sv.reshape(1, -1)
        return(vector)

def GetTintJson (sentence):
    response = requests.post(url, data={'text': sentence.rstrip()})
    return(response)
##
# Convert to string keeping encoding in mind...
##
def to_string(s):
    try:
        return str(s)
    except:
        #Change the encoding type if needed
        return s.encode('utf-8')



def reduce_item(key, value):
    global reduced_item

    #Reduction Condition 1
    if type(value) is list:
        i=0
        for sub_item in value:
            reduce_item(key+'_'+to_string(i), sub_item)
            i=i+1

    #Reduction Condition 2
    elif type(value) is dict:
        sub_keys = value.keys()
        for sub_key in sub_keys:
            reduce_item(key+'_'+to_string(sub_key), value[sub_key])

    #Base Condition
    else:
        reduce_item[to_string(key)] = to_string(value)


if __name__ == "__main__":

    url = 'http://dh-server.fbk.eu:19003/simp-engines/tae/simpform'

    print("Loading FT model")
    model_ft  = fasttext.load_model('cc.it.300.bin')


    print("Processing data")

    if len(sys.argv) != 4:
        print ("\nUsage: python json_to_csv.py <node> <json_in_file_path> <csv_out_file_path>\n")
    else:
        #Reading arguments
        node = sys.argv[1]
        json_file_path = sys.argv[2]
        outFileName = sys.argv[3]

        fp = open(json_file_path, 'r')
        fileOut = open (outFileName, 'w')

        json_value = fp.read()
        raw_data = json.loads(json_value)
        fp.close()

        for i in raw_data:

                textDescription = i["lenght"]
                label = i['results']

                myEmbeddings = ExtractVectors(textDescription, False)
                myEmbeddingsString=""
                counter=1
                for v in myEmbeddings[0]:
                    myEmbeddingsString = myEmbeddingsString+" "+str(counter)+":"+str(v)
                    counter = counter +1

                # myEmbeddingsString = str(myEmbeddings).replace("[","")
                # myEmbeddingsString = myEmbeddingsString.replace("]","")

                if label =="Good":
                    numLabel = "1"
                else:
                    numLabel = "0"

                fileOut.write(numLabel+" "+myEmbeddingsString+"\n")

        fileOut.close()
