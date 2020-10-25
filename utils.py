import pandas as pd 
import numpy as np 
from nltk.tokenize import sent_tokenize
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

def extract_polarity(data): 
    dataset_list = []
    sid = SentimentIntensityAnalyzer() 
    patient_list = list() 
    neg,neu,pos = list(),list(),list()
    for sent in sent_tokenize(data):  
        tmp = sid.polarity_scores(sent) 
        neg.append(tmp['neg'])
        neu.append(tmp['neu'])
        pos.append(tmp['pos']) 
    data = [np.mean(neg),np.mean(neu),np.mean(pos)]
    return data 