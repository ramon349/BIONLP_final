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
    id = str(tweet['id'])
    content = tweet['text'] 
    sentimet =  extract_polarity(content)  
    fin_senti = sentiments[np.argmax(sentimet)]
    return (id,content,fin_senti) 

def main_three():
    import services.mongo_setup as mongo_setup 
    import services.data_service as data_service
    mongo_setup.global_init() 
    sample_tweet= data_service.find_tweet_by_mood(mood='neu').first()
    print(f"For tweet {sample_tweet.tweetID}") 
    print(f"Sentiment was {sample_tweet.mood} ")
    print(f"TEXT:  {sample_tweet.text}")
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
        print(f"Done adding stauts {i}-----")
if __name__ == "__main__":  
    main_two()