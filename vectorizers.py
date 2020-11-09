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
from gary_utils import * 



class myVectorizer():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['Text']
        
        textLens=[]
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            textLens.append(len(word_tokenize(tr)))
            
        
        self.textVectorizer.fit(texts_preprocessed)
        self.clustervectorizer.fit(clusters) 
        
        self.maxs['len']=max(textLens)
        self.mins['len']=min(textLens)
    
    def transform(self, rows):
        unprocessedTexts=rows['Text']
        
        texts_preprocessed = []
        clusters=[]
        textLens=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            textLens.append(len(word_tokenize(tr)))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        cluster_vectors = self.clustervectorizer.transform(clusters).toarray()

        data_vectors = np.concatenate((data_vectors, cluster_vectors), axis=1)
        
        textLensNorm=getNormalizedList(textLens, self.maxs['len'], self.mins['len'])
        data_vectors = np.concatenate((data_vectors, np.array([textLensNorm]).T), axis=1)
        
        return data_vectors
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)

def text2pred():
    model = pkl.load(open('adaForest10_22_2020.pickle','rb'))
    def pred(txt): 
        txt = preprocess_text(txt)
        inputDF=pd.DataFrame({'Text':[txt]})
        pred=model.predict(inputDF)[0]
        return pred
    return pred

if __name__=="__main__":
    import pickle as pkl 
    classy = text2pred(" I have breast cancer")
    print(classy("pokemon is a cool show"))