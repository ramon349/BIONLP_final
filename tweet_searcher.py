import requests
import pandas as pd
import json
import ast
import yaml
from vectorizers import myVectorizer , text2pred
sentiments = ['noReports','selfReport']


class TweetAPI: 
    def __init__(self,config_path="config.yaml"):
        token_file = self.process_yaml(config_path)
        self.bearer_token = create_bearer_token(token_file) 
    def create_twitter_url(self,queryItem,max_results=100):
        q = "q={}&max_results={}".format(queryItem,max_results)
        url = "https://api.twitter.com/1.1/search/tweets.json?{}&tweet_mode=extended".format(q)
        return url
    def twitter_auth_and_connect(self,url):
        headers = {"Authorization": "Bearer {}".format(self.bearer_token)}
        response = requests.request("GET", url, headers=headers)
        return response.json() 
    def search(self,query):
        search_url = self.create_twitter_url(queryItem=query)
        return self.twitter_auth_and_connect(search_url)
    def process_yaml(self,path):
        with open(path) as file:
            return yaml.safe_load(file)
def create_bearer_token(data):
    return data["access_token"]



def main(): 
    ramenSearcher = TweetAPI()
    output = ramenSearcher.search("breast cancer")
    pd_list = list() 
    clasi = text2pred() 
    for e in output['statuses']:
        print(e)
        txt:str  = e['full_text']
        breakpoint() 
       
        id = e['id']
        senti = sentiments[clasi(txt) ]
        d  = pd.DataFrame.from_dict({'id':id,'text':txt,'report':senti},orient='index').T
        pd_list.append(d)
    other_data =  pd.concat(pd_list)
    other_data.to_csv('searched_tweets.csv')
if __name__ == "__main__":
    main()