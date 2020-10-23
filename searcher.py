import requests
import pandas as pd
import json
import ast
import yaml




class TweetAPI: 
    def __init__(self,config_path="config.yaml"):
        token_file = self.process_yaml(config_path)
        self.bearer_token = create_bearer_token(token_file) 
    def create_twitter_url(self,queryItem,max_results=100):
        q = "q={}&max_results={}".format(queryItem,max_results)
        url = "https://api.twitter.com/1.1/search/tweets.json?{}".format(q)
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
    output = ramenSearcher.search("digimon movie")
    print(output)


if __name__ == "__main__":
    main()
