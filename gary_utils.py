import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import word_tokenize
import pickle as pkl
import re
import nltk

word_clusters = {}
st = stopwords.words('english')
def preprocess_text(raw_text):

    # Replace/remove username
    raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing
    words=[]
    for w in raw_text.lower().split():
        if not w in st and not w in ['.',',', '[', ']', '(', ')']:
            words.append(w)
            
    return (" ".join(words))

def loadwordclusters():
    infile = open('./50mpaths2',  "r", encoding="utf-8")
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)


def normalize(value, maxOfList, minOfList):
    return (value - minOfList) / (maxOfList - minOfList)
def getNormalizedList(values, maxOfList, minOfList):
    ret = []
    for value in values:
        ret.append(normalize(value, maxOfList, minOfList))
        
    return ret  

loadwordclusters()

