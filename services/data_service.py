from  mongoOBJS.mydata import tweetData
#this funciton handles adding an elements to the database 
def create_finding(id:str, text:str,report:str) -> tweetData:
    tweet = tweetData()
    tweet.tweetID = id
    tweet.text = text
    tweet.report = report
    tweet.save()
    return tweet
#searching the database for a tweet whose status equals that of the  input 
def find_tweet_by_mood(report:str) -> tweetData:
    #can use this to see if the tweet id has already been added 
    tweet = tweetData.objects().filter(report=report)
    return tweet