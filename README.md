# BIONLP_final


## Streaming Data:

# 1 Getting tweets 

Use play_tweeepy.py to start the process. 
Ramon has an enviroment set up to load twitter api keys. 
Code defines an implementation of MyStreamListener 
  - requries that we specify an on_status file that pulls tweets 
  - tweet_process: wil ltake the tweet and try to extract the self report classification 

The tweet is stored into the mondo db service with the following tags (id,content,sentiment)

What does mydata.py do? 
  it provides a wrapper for mongodb so we can upload data 

# 2 classifier 

Project task is 
 1. Ramon Gary first half 
 2. Thaigo and Yusen  do second part
   2.a TBA  final task assignment 

# todo 
1.  Gary will do the classifier 
2.  Ramon will work on streaming component using dummy classifier 
3.  Annotate the first 100 tweets
4. IAAA (Gary)
