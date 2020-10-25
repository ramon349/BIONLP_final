import tweepy 
import os
import numpy as np 
from utils import extract_polarity
import services.mongo_setup as mongo_setup  #this sets up mongodb 
import services.data_service as data_service #this handles adding elements to the database 
sentiments = ['neg','neu','pos'] 

class MyStreamListener(tweepy.StreamListener):
    """ This object handles reading twitters stream 
    """
    def on_status(self, status):
        """  Stream in tweets and add them to the database 

        """
        rep = status._json
        (id,cont,fin_senti)= self.tweet_process(rep) #this does the sentiment analysis j
        data_service.create_finding(id,cont,fin_senti) #this adds to the datbase 
        return (True,)
    def on_error(self,status): 
        return status
    def tweet_process(self,tweet): 
        """  takes a tweet runs it through a simple sentiment analyzer. returns the id and content in a dataframe 
        """
        id = str(tweet['id'])
        content = tweet['text'] 
        sentimet =  extract_polarity(content)  
        fin_senti = sentiments[np.argmax(sentimet)]
        return (id,content,fin_senti) 
def main():
    #initialize using twitter api. these are my enviroment keys 
    auth = tweepy.OAuthHandler(os.environ['API_KEY'], os.environ['API_SECRET_KEY'])
    auth.set_access_token(os.environ['T_ACCESS'], os.environ['T_SECRET'])
    api = tweepy.API(auth)
    myListen = MyStreamListener() 
    myStream = tweepy.Stream(auth=api.auth,listener=myListen)
    myStream.filter(track=['debate']) #term we'll be tracking. Using debate  currently for the sake of simplicity 
if __name__=="__main__":
    mongo_setup.global_init()
    main()