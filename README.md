# BIONLP_final


# Requirements 
1) Main requirements for the package are  sklearn,pandas, tweepy.
2) You will need to have your twitter api keys and tokens in a config.yaml file 
  2.a) it is also possible to pass in the path to your config file as a parameter 
  2.b) API_KEY,API_SECRET_KEY,T_ACCESS,T_SECRET should be deifned in config
# 1 Getting tweets 

Use play_tweeepy.py to start the process. 

The tweet is stored into the mondo db service with the following tags (id,content,selfReportStatus) 
Services folder provides mongo_setup and data_service classes needed by mongo db 

mongo_setup.py: provies script to initialize mongodb and prep for adding data
data_service.py: Provies two key functionalities. 1) adding element sto database 2) filter elements in database by report type 

# 2 classifier 
proj2.ipynb: is where the model exploration occured. Here we can re-run all of the classification experiemtn's done by gary. I.e the ablation study and grid search. 

proj2.py is a copy of the notbeook meant to be run as a pyton module. this is used for importing processing code meant to be used by the classifier. This is neeeded to succesffully load the pickled pipeline 


# How to use the non-bert classifier
open annotateTheUnannotatedDataset

copy and paste code from it

the pipeline has a few dependencies, like the vectorizer that you see in the notebook

if you're using jupyter notebook, you can get away with running
%run annotateTheUnannotatedDataset.ipynb

in your notebook. This will run the notebook as if you had imported the whole damn thing


# addiitonal scripts 
  text_consumer.py: Is used to read all the classes of interest in the database and produce speerate csv files. 
    1) there is an additional check added due to the issue of some tweets from an individual retweeting. 
    2) the name list is the database name.
