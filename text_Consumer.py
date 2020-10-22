import os 
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize
from utils import extract_polarity
import json 
import time 
import numpy as np 
import pandas as pd 
sentiments = ['neg','neu','pos']

def tweet_process(tweet): 
    """  takes a tweet runs it through a simple sentiment analyzer. returns the id and content in a dataframe 

    """
    id = tweet['id']
    content = tweet['text'] 
    sentimet =  extract_polarity(content)  
    fin_senti = sentiments[np.argmax(sentimet)]
    return pd.DataFrame.from_dict({'id':[id],'content':[content]},orient='columns'),fin_senti
if __name__ == "__main__": 
    sampleDoc = eval(open('example.json','r').read() )
    aggregators ={'neg':list(),'neu':list(),'pos':list()}
    print("Aggregating sentiments")
    for e in sampleDoc['statuses']: 
        data,clasi = tweet_process(e)
        aggregators[clasi].append(data) 
    print("Writting  csv's" )
    for e in aggregators.keys():
        if len(aggregators[e]) > 0 :
            tmp_data = pd.concat(aggregators[e]).to_csv(f'{e}_data.csv',index=False)