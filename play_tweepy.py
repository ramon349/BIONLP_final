import tweepy 
import os

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        breakpoint()
        print(status.text)
        return (True,)
    def on_error(self,status):
        return status

def main():
    auth = tweepy.OAuthHandler(os.environ['API_KEY'], os.environ['API_SECRET_KEY'])
    auth.set_access_token(os.environ['T_ACCESS'], os.environ['T_SECRET'])
    api = tweepy.API(auth)
    myListen = MyStreamListener() 
    myStream = tweepy.Stream(auth=api.auth,listener=myListen)
    myStream.filter(track=['debate'])

if __name__=="__main__":
    main()