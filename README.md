# BIONLP_final

Project task is 
 1. Ramon Gary first half 
 2. Thaigo and Yusen  do second part
   2.a TBA  final task assignment 

# todo 
1.  Gary will do the classifier 
2.  Ramon will work on streaming component using dummy classifier 

# How to use the non-bert classifier
use 
filename = 'adaForest11_9_2020.pickle'
model = pickle.load(open(filename, 'rb'))
model.predict(test_rows)

where test_rows is a pandas dataframe with "Text" as one of the column names.
it outputs either a 1  (self report) or 0 (not self report)
