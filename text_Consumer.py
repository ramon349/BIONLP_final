import os 
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize
from utils import extract_polarity
import json 
import time 
import numpy as np 
import pandas as pd 
sentiments = ['noReport','selfReport']

def tweet_process(tweet): 
    """  takes a tweet runs it through a simple sentiment analyzer. returns the id and content in a dataframe 

    """
    id = str(tweet['id'])
    content = tweet['text'] 
    breakpoint()
    sentimet =  extract_polarity(content)  
    fin_senti = sentiments[np.argmax(sentimet)]
    return (id,content,fin_senti) 

def main_three(reportType,out_file):
    import services.mongo_setup as mongo_setup 
    import services.data_service as data_service
    mongo_setup.global_init() 
    sample_tweet= data_service.find_tweet_by_mood(report=reportType).all()
    rep_list = list() 
    print('Start appending')
    for e in sample_tweet: 
        text=e.text
        report=e.report 
        id = e.tweetID
        if text.find("@nadidoesart")>=1: 
            print('SKIP')
            continue 
        d = pd.DataFrame.from_dict({'id':id,'text':text,'report':report},orient='index').T
        rep_list.append(d)
    print('Done appending')
    data = pd.concat(rep_list)
    data.to_csv(out_file)
    
#main for when testing mongo for now 
def main_two():
    #import services.mongo_setup as mongo_setup 
    #import services.data_service as data_service
    #mongo_setup.global_init() 
    sampleDoc = eval(open('example.json','r').read() ) # didn't write properly loading as dict 
    aggregators ={'neg':list(),'neu':list(),'pos':list()}
    print("Aggregating sentiments")
    for i,e in enumerate(sampleDoc['statuses']): 
        print(f"Adding status {i}-----")
        id,content,fin_senti = tweet_process(e) 
        #data_service.create_finding(id,content,fin_senti)
        print(f"Done adding status {i}-----")
if __name__ == "__main__":  
    print("Working on self reports")
    main_three('selfReport','selfReps_before_nov_9.csv')
    print("Working on nonSelfReports")
    main_three('noReport','noReps_before_nov_9.csv')