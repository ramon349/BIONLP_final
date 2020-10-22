import pandas as pd 
import numpy as np 
from nltk.tokenize import sent_tokenize
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

def extract_polarity(data): 
    dataset_list = []
    sid = SentimentIntensityAnalyzer() 
    for i,submission in enumerate(data): 
        patient_list = list() 
        neg,neu,pos = list(),list(),list()
        for sent in sent_tokenize(submission):  
            tmp = sid.polarity_scores(sent) 
            neg.append(tmp['neg'])
            neu.append(tmp['neu'])
            pos.append(tmp['pos'])
        dataset_list.append(pd.DataFrame(data={'neg':np.mean(neg),'neu':np.mean(neu),'pos':np.mean(pos)},index=[i])) 
    output = pd.concat(dataset_list).to_numpy() 
    return output 