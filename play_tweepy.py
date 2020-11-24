import tweepy 
import services.mongo_setup as mongo_setup  #this sets up mongodb 
import services.data_service as data_service #this handles adding elements to the database 
from gary_utils import * 
from vectorizers import myVectorizer,  text2pred
import yaml 
sentiments = ['noReport','selfReport']

class MyStreamListener(tweepy.StreamListener):
    """ Listener object required for reading twitter data stream  
    """
    def __init__(self,classy):
        """Takes a classifier object as input. 
        classy: is an sklearn pipleine object that does text preprocessing
         and classificaiton. 
        """
        super().__init__()
        self.classy = classy
    def on_status(self, status):
        """  Stream  tweets and add them to the database 
            status: tweepy object containing a tweets data 
        """
        rep = status._json
        (id,cont,fin_senti)= self.tweet_process(rep) #this does the sentiment analysis j
        data_service.create_finding(id,cont,fin_senti) #this adds to the datbase 
        return (True,)
    def on_error(self,status): 
        return status
    def tweet_process(self,tweet): 
        """  takes a tweet runs it through a simple analyzer. returns the id and content in a dataframe 
        """
        id = str(tweet['id'])
        try: 
            #some tweets have longer text. Attempt to extract it 
            content = tweet['extended_tweet']['full_text']
        except KeyError: 
            content = tweet['text']
        output =sentiments[self.classy(content)]
        return (id,content,output) 
def main():
    #initialize using twitter api. these are my enviroment keys 
    f_path = 'config.yaml'
    secrets= load_credentials(f_path)
    auth = tweepy.OAuthHandler(secrets['API_KEY'], secrets['API_SECRET_KEY'])
    auth.set_access_token(secrets['T_ACCESS'], secrets['T_SECRET'])
    api = tweepy.API(auth)
    classi = text2pred()
    myListen = MyStreamListener(classy=classi) 
    myStream = tweepy.Stream(auth=api.auth,listener=myListen)
    #each element on the search list is a single item .
    myStream.filter(track=['breast cancer', 'breast lump','breast pain'])

def load_credentials(f_path): 
    with open(f_path) as f: 
        return yaml.safe_load(f)
if __name__=="__main__":
    mongo_setup.global_init()
    main()