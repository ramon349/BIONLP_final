#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
import sys
!{sys.executable} -m pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html --user
#"""


# In[26]:


import torch
torch.__version__


# In[27]:


torch.cuda.is_available()


# In[28]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import word_tokenize
import re
import nltk

st = stopwords.words('english')
stemmer = PorterStemmer()

def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df


def preprocess_text(raw_text):

    # Replace/remove username
    raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing
    words=[]
    for w in raw_text.lower().split():
        if not w in st and not w in ['.',',', '[', ']', '(', ')']:
            words.append(w)
            
    return (" ".join(words))


# In[29]:


#Load the data
f_path = './Breast Cancer(Raw_data_2_Classes).csv'
data = loadDataAsDataFrame(f_path)

texts = data['Text']
classes = data['Class']
ids = data['ID']

#PREPROCESS THE DATA
texts_preprocessed=[preprocess_text(txt) for txt in texts]
data['preprocessed_texts']=texts_preprocessed

data


# In[ ]:





# In[30]:


data.iloc[3]['Text']


# In[31]:


from collections import Counter

Counter(data['Class'])


# In[32]:




word_clusters = {}

def loadwordclusters():
    infile = open('./50mpaths2',  "r", encoding="utf-8")
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

loadwordclusters()

class myVectorizer():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['Text']
        
        textLens=[]
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            textLens.append(len(word_tokenize(tr)))
            
        
        self.textVectorizer.fit(texts_preprocessed)
        self.clustervectorizer.fit(clusters) 
        
        self.maxs['len']=max(textLens)
        self.mins['len']=min(textLens)
    
    def transform(self, rows):
        unprocessedTexts=rows['Text']
        
        texts_preprocessed = []
        clusters=[]
        textLens=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            textLens.append(len(word_tokenize(tr)))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        cluster_vectors = self.clustervectorizer.transform(clusters).toarray()

        data_vectors = np.concatenate((data_vectors, cluster_vectors), axis=1)
        
        textLensNorm=getNormalizedList(textLens, self.maxs['len'], self.mins['len'])
        data_vectors = np.concatenate((data_vectors, np.array([textLensNorm]).T), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)


# In[ ]:





# In[33]:


def grid_search_hyperparam_space(params, pipeline, folds, training_texts, training_classes):#folds, x_train, y_train, x_validation, y_validation):
        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='f1_macro',n_jobs=-1)
        grid_search.fit(training_texts, training_classes)
        return grid_search


# In[ ]:





# ## Split the data

# In[10]:


from sklearn.model_selection import train_test_split

training_set_size = int(0.8*len(data))

X=data
y=data['Class'].tolist()

training_rows, test_rows, training_classes, test_classes = train_test_split(
    X, y, train_size=training_set_size, random_state=42069)


# In[11]:



def normalize(value, maxOfList, minOfList):
    return (value - minOfList) / (maxOfList - minOfList)
    
def getNormalizedList(values, maxOfList, minOfList):
    ret = []
    for value in values:
        ret.append(normalize(value, maxOfList, minOfList))
        
    return ret  


# In[12]:


from sklearn.metrics import precision_recall_fscore_support as prf1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
import random

def bulkEval(predictions_test, test_classes, bs=False):
    print ("Accuracy\t", acc(predictions_test,test_classes))
    macro=f1(predictions_test,test_classes, average='macro')
    micro=f1(predictions_test,test_classes, average='micro')
    print ("F1 Macro\t", macro)
    print ("F1 Micro\t", micro)
    print("Confusion Matrix")
    print(confusion_matrix(test_classes, predictions_test, labels=[1,0], normalize='true'))

    #bootstrap it
    if bs:
        f1s=[]
        for iteration in range(1000):
            resampleIndexes=random.choices(range(len(predictions_test)), k=1000)
            resamplePreds=[predictions_test[i] for i in resampleIndexes]
            resampleTrueClasses=[test_classes[i] for i in resampleIndexes]
            thisF1=f1(resamplePreds,resampleTrueClasses, average='macro')
            f1s.append(thisF1)

        print("Bootstrapping 95% confidence interval:")
        interval=np.percentile(f1s, [2.5, 97.5])
        print(interval)

    print("\t****************************************\n")

    #entry={"Classifier": clf, "F1 Macro":macro, "F1 Micro":micro,
    #          "Confidence Interval":interval}
    #f1df=f1df.append(entry, ignore_index=True)


# In[13]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# ## Guess 0 Classifier
# The Stupidest classifier possible

# In[14]:


class stupidClassifier():
    def fit(self, X,y):
        doNothing=True
        
    def predict(self, X):
        ret = []
        for xi in X.iterrows():
            ret.append(0)
        return ret


# In[15]:


#CLASSIFIER
clf=stupidClassifier()

#CLASSIFY AND EVALUATE 
predictions_test = clf.predict(test_rows)
print('Performance on held-out test set ... :')

bulkEval(predictions_test,test_classes, bs=True)


# ## GNB baseline
# 

# In[16]:



vectorizer = myVectorizer()

#CLASSIFIER
gnb_classifier = GaussianNB()
grid_params = {}

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',gnb_classifier)])

#SEARCH HYPERPARAMETERS
folds = 5
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

bulkEval(predictions_test,test_classes, bs=True)


# In[ ]:





# ## simple transformer RoBERTa

# In[25]:


from simpletransformers.classification import ClassificationModel

model_args={'overwrite_output_dir':True}

# Create a TransformerModel
model = ClassificationModel('roberta', 'roberta-base', use_cuda=False, args=model_args)
#model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, args=model_args)

#change our data into a format that simpletransformers can process
training_rows['text']=training_rows['Text']
training_rows['labels']=training_rows['Class']
test_rows['text']=test_rows['Text']
test_rows['labels']=test_rows['Class']

# Train the model
model.train_model(training_rows)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_rows)

print("f1 score")
precision=result['tp'] / (result['tp'] + result['fp'])
recall=result['tp'] / (result['tp'] + result['fn'])
f1score= 2 * precision * recall / (precision + recall)
print(f1score)


# # Optimize Parameters

# ## SVM classifier
# 
# Best hyperparameters:
# {'svm_classifier__C': 4, 'svm_classifier__kernel': 'rbf'}

# In[ ]:


vectorizer = myVectorizer()

#CLASSIFIER
svm_classifier = svm.SVC(gamma='scale')

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('svm_classifier',svm_classifier)])

grid_params = {
     'svm_classifier__C': [0.25,1,4,16,64],
     'svm_classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

print('Best hyperparameters:')
print(grid.best_params_)


#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

print(accuracy_score(predictions_test,test_classes))


# In[ ]:


svm_classifier = svm.SVC(gamma='scale')

pipeline = Pipeline(steps = [('vec',vectorizer),('svm_classifier',svm_classifier)])

grid_params = {
     'svm_classifier__C': [4],
     'svm_classifier__kernel': ['rbf'],
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

print(bulkEval(predictions_test,test_classes))


# ## Random Forest
# 
# Best hyperparameters:
# {'classifier__n_estimators': 5}
# 
# 

# In[ ]:


vectorizer = myVectorizer()

rf = RandomForestClassifier()

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',rf)])
#pipeline ensures vectorization happens in each fold of grid search 
#(you could code the entire process manually for more flexibility)

grid_params = {
     'classifier__n_estimators': np.arange(5,60,5)
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

print('Best hyperparameters:')
print(grid.best_params_)

print('Optimal n found:', grid.best_params_['classifier__n_estimators'])

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

print(accuracy_score(predictions_test,test_classes))


# ## KNN
# 
# Best hyperparameters:
# {'classifier__n_neighbors': 3}
# 
# 

# In[ ]:





# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

vectorizer = myVectorizer()

clf= KNeighborsClassifier()

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])
#pipeline ensures vectorization happens in each fold of grid search 
#(you could code the entire process manually for more flexibility)

grid_params = {
     'classifier__n_neighbors': np.arange(1,20,1),
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

print('Best hyperparameters:')
print(grid.best_params_)

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

print(accuracy_score(predictions_test,test_classes))


# ## Neural Network
# 
# Is it really data science if there isn't a neural network somewhere?
# 
# Best hyperparameters:
# {'classifier__hidden_layer_sizes': (11,)}
# 

# In[ ]:


from sklearn.neural_network import MLPClassifier

vectorizer = myVectorizer()

clf= MLPClassifier()

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])
#pipeline ensures vectorization happens in each fold of grid search 
#(you could code the entire process manually for more flexibility)

#we'll just use one hidden layer
layerParams=[]
for n in range(1,101, 10):
    layerParams.append(tuple([n]))
    
grid_params = {
     'classifier__hidden_layer_sizes': layerParams,
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

print('Best hyperparameters:')
print(grid.best_params_)

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

print(accuracy_score(predictions_test,test_classes))


# ## adaboost
# 
# Best hyperparameters:
# {'classifier__base_estimator__max_depth': 3, 'classifier__base_estimator__n_estimators': 30, 'classifier__n_estimators': 50}

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=420)

#SIMPLE PIPELINE
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])
#pipeline ensures vectorization happens in each fold of grid search 
#(you could code the entire process manually for more flexibility)

#we'll just use one hidden layer
layerParams=[]
for n in range(1,101, 10):
    layerParams.append(tuple([n]))
    
grid_params = {
     "classifier__base_estimator__n_estimators":range(10, 51, 20),
     "classifier__base_estimator__max_depth":range(3),
    "classifier__n_estimators":range(10, 51, 20)
}

#SEARCH HYPERPARAMETERS
folds = 2
grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

print('Best hyperparameters:')
print(grid.best_params_)

#CLASSIFY AND EVALUATE 
predictions_test = grid.predict(test_rows)
print('Performance on held-out test set ... :')

bulkEval(predictions_test,test_classes)


# ## Now evaluate them all

# In[ ]:


from sklearn.metrics import confusion_matrix
import random

"""
Confusion matrix whose i-th row and j-th column entry indicates the number of samples 
with true label being i-th class and prediced label being j-th class.
"""

gnb = GaussianNB()
svmc = svm.SVC(C=4, kernel='rbf', gamma='scale', probability=True)
rf = RandomForestClassifier(n_estimators=5)
knn=KNeighborsClassifier(n_neighbors=3)
nn=MLPClassifier(hidden_layer_sizes=(11,))
en=VotingClassifier(estimators=[('SVM', svmc), ('RF', rf), 
                                ("KNN", knn), ("NN", nn)], 
                                      voting='soft')

f1df=pd.DataFrame()

for clf in [gnb, svmc, rf, knn, nn, en]:
    vectorizer = myVectorizer()

    #SIMPLE PIPELINE
    pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])

    grid_params = {}
    #SEARCH HYPERPARAMETERS
    folds = 2
    grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

    print('Best hyperparameters:')
    print(grid.best_params_)

    #CLASSIFY AND EVALUATE 
    predictions_test = grid.predict(test_rows)
    
    print("Classifier\t", clf)

    print ("Accuracy\t", acc(predictions_test,test_classes))
    macro=f1(predictions_test,test_classes, average='macro')
    micro=f1(predictions_test,test_classes, average='micro')
    print ("F1 Macro\t", macro)
    print ("F1 Micro\t", micro)
    print("Confusion Matrix")
    print(confusion_matrix(test_classes, predictions_test, labels=['CoM', 'Other'], normalize='true'))
    
    #bootstrap it
    f1s=[]
    for iteration in range(1000):
        resampleIndexes=random.choices(range(len(predictions_test)), k=len(predictions_test))
        resamplePreds=[predictions_test[i] for i in resampleIndexes]
        resampleTrueClasses=[test_classes[i] for i in resampleIndexes]
        thisF1=f1(resamplePreds,resampleTrueClasses, average='macro')
        f1s.append(thisF1)
        
    print("Bootstrapping 95% confidence interval:")
    interval=np.percentile(f1s, [2.5, 97.5])
    print(interval)
    
    print("\t****************************************\n")
    
    entry={"Classifier": clf, "F1 Macro":macro, "F1 Micro":micro,
              "Confidence Interval":interval}
    f1df=f1df.append(entry, ignore_index=True)
    
f1df


# ## The winner: Adaboost RF

# ## PICKLE IT

# In[ ]:


import sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=30, max_depth=3), n_estimators=50)
vectorizer=myVectorizer()
pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])


# In[ ]:


pipeline.fit(data, data['Class'].tolist())


# In[ ]:


predictions_test = pipeline.predict(test_rows)
bulkEval(predictions_test, test_classes)


# In[35]:


import pickle

# save the model to disk
#filename = 'adaForest10_22_2020.pickle'
#pickle.dump(pipeline, open(filename, 'wb'))


# # Just give me a function that takes an input and spits out a result

# In[40]:


filename = 'adaForest10_22_2020.pickle'
model = pickle.load(open(filename, 'rb'))

def textToPred(txt):
    inputDF=pd.DataFrame({'Text':[txt]})
    pred=model.predict(inputDF)[0]
    return pred

textToPred("My micro pen is not a self-report")


# In[ ]:





# ## Ablation Study

# In[ ]:



#remove fall duration
class ablation1():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.cnumclustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.loccnumClusterVectorizer= CountVectorizer(ngram_range=(1,1), max_features=1000)
        
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['fall_description']
        
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            
        allcnums=[]
        allCNumLists=rows['CNums']
        for cnumList in allCNumLists:
            allcnums.append(cnumList)
        
        self.textVectorizer.fit(texts_preprocessed)
        self.clustervectorizer.fit(clusters)
        self.cnumclustervectorizer.fit(allcnums)  
        
        #fall location
        allLoccnums=[]
        allLocCNumLists=rows['LocCNums']
        for cnumList in allLocCNumLists:
            allLoccnums.append(cnumList)
            
        self.loccnumClusterVectorizer.fit(allLoccnums)  
            
    
    def transform(self, rows):
        unprocessedTexts=rows['fall_description']
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        cluster_vectors = self.clustervectorizer.transform(clusters).toarray()

        data_vectors = np.concatenate((data_vectors, cluster_vectors), axis=1)
        
        allcnums=rows['CNums']
        cnum_cluster_vectors = self.cnumclustervectorizer.transform(allcnums).toarray()
        allloccnums=rows['LocCNums']
        loc_cnum_cluster_vectors = self.loccnumClusterVectorizer.transform(allloccnums).toarray()
        
        data_vectors = np.concatenate((data_vectors, cnum_cluster_vectors), axis=1)
        data_vectors = np.concatenate((data_vectors, loc_cnum_cluster_vectors), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)

#remove metamap tags of locations
class ablation2():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.cnumclustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['fall_description']
        
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            
        allcnums=[]
        allCNumLists=rows['CNums']
        for cnumList in allCNumLists:
            allcnums.append(cnumList)
        
        self.textVectorizer.fit(texts_preprocessed)
        self.clustervectorizer.fit(clusters)
        self.cnumclustervectorizer.fit(allcnums)  
        
        
        #get ready to normalize all the other features
        for feature in training_rows.columns:
            values=training_rows[feature]
            featureType=type(values[0])

            if not featureType==str or featureType==int:
                self.maxs[feature]=max(values)
                self.mins[feature]=min(values)
            
        
    
    def transform(self, rows):
        unprocessedTexts=rows['fall_description']
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        cluster_vectors = self.clustervectorizer.transform(clusters).toarray()

        data_vectors = np.concatenate((data_vectors, cluster_vectors), axis=1)
        
        allcnums=rows['CNums']
        cnum_cluster_vectors = self.cnumclustervectorizer.transform(allcnums).toarray()

        data_vectors = np.concatenate((data_vectors, cnum_cluster_vectors), axis=1)

        
        #tack on all the other numeric features
        for feature in ['duration',]:
            values=rows[feature]
            normValues = np.array([getNormalizedList(values, self.maxs[feature], self.mins[feature])])
            data_vectors=np.concatenate((data_vectors, normValues.T), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)

#remove metamap tags of fall descriptions
class ablation3():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.loccnumClusterVectorizer= CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['fall_description']
        
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
            
        
        self.textVectorizer.fit(texts_preprocessed)
        self.clustervectorizer.fit(clusters)
        
        #fall location
        allLoccnums=[]
        allLocCNumLists=rows['LocCNums']
        for cnumList in allLocCNumLists:
            allLoccnums.append(cnumList)
            
        self.loccnumClusterVectorizer.fit(allLoccnums)  
        
        
        #get ready to normalize all the other features
        for feature in training_rows.columns:
            values=training_rows[feature]
            featureType=type(values[0])

            if not featureType==str or featureType==int:
                self.maxs[feature]=max(values)
                self.mins[feature]=min(values)
            
        
    
    def transform(self, rows):
        unprocessedTexts=rows['fall_description']
        texts_preprocessed = []
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            clusters.append(getclusterfeatures(tr))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        cluster_vectors = self.clustervectorizer.transform(clusters).toarray()

        data_vectors = np.concatenate((data_vectors, cluster_vectors), axis=1)
        
        allloccnums=rows['LocCNums']
        loc_cnum_cluster_vectors = self.loccnumClusterVectorizer.transform(allloccnums).toarray()
        
        data_vectors = np.concatenate((data_vectors, loc_cnum_cluster_vectors), axis=1)

        #tack on all the other numeric features
        for feature in ['duration',]:
            values=rows[feature]
            normValues = np.array([getNormalizedList(values, self.maxs[feature], self.mins[feature])])
            data_vectors=np.concatenate((data_vectors, normValues.T), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)

#remove 50mpaths clusters of fall descriptions
class ablation4():
    def __init__(self):
        self.textVectorizer=CountVectorizer(ngram_range=(1, 3), max_features=10000)
        self.cnumclustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.loccnumClusterVectorizer= CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['fall_description']
        
        texts_preprocessed = []
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
            
        allcnums=[]
        allCNumLists=rows['CNums']
        for cnumList in allCNumLists:
            allcnums.append(cnumList)
        
        self.textVectorizer.fit(texts_preprocessed)
        self.cnumclustervectorizer.fit(allcnums)  
        
        #fall location
        allLoccnums=[]
        allLocCNumLists=rows['LocCNums']
        for cnumList in allLocCNumLists:
            allLoccnums.append(cnumList)
            
        self.loccnumClusterVectorizer.fit(allLoccnums)  
        
        #get ready to normalize all the other features
        for feature in training_rows.columns:
            values=training_rows[feature]
            featureType=type(values[0])

            if not featureType==str or featureType==int:
                self.maxs[feature]=max(values)
                self.mins[feature]=min(values)
            
        
    
    def transform(self, rows):
        unprocessedTexts=rows['fall_description']
        texts_preprocessed = []
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            texts_preprocessed.append(preprocess_text(tr))
        
        data_vectors = self.textVectorizer.transform(texts_preprocessed).toarray()
        
        allcnums=rows['CNums']
        cnum_cluster_vectors = self.cnumclustervectorizer.transform(allcnums).toarray()
        allloccnums=rows['LocCNums']
        loc_cnum_cluster_vectors = self.loccnumClusterVectorizer.transform(allloccnums).toarray()
        
        data_vectors = np.concatenate((data_vectors, cnum_cluster_vectors), axis=1)
        data_vectors = np.concatenate((data_vectors, loc_cnum_cluster_vectors), axis=1)

        
        #tack on all the other numeric features
        for feature in ['duration',]:
            values=rows[feature]
            normValues = np.array([getNormalizedList(values, self.maxs[feature], self.mins[feature])])
            data_vectors=np.concatenate((data_vectors, normValues.T), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)
    
#remove the n-grams
class ablation5:
    def __init__(self):
        self.clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.cnumclustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=1000)
        self.loccnumClusterVectorizer= CountVectorizer(ngram_range=(1,1), max_features=1000)
        
        #for normaluzation
        self.maxs={}
        self.mins={}
    
    def fit(self, rows, y=None):
        
        #fall description
        unprocessedTexts=rows['fall_description']
        
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            clusters.append(getclusterfeatures(tr))
            
        allcnums=[]
        allCNumLists=rows['CNums']
        for cnumList in allCNumLists:
            allcnums.append(cnumList)
        
        self.clustervectorizer.fit(clusters)
        self.cnumclustervectorizer.fit(allcnums)  
        
        #fall location
        allLoccnums=[]
        allLocCNumLists=rows['LocCNums']
        for cnumList in allLocCNumLists:
            allLoccnums.append(cnumList)
            
        self.loccnumClusterVectorizer.fit(allLoccnums)  
        
        
        #get ready to normalize all the other features
        for feature in training_rows.columns:
            values=training_rows[feature]
            featureType=type(values[0])

            if not featureType==str or featureType==int:
                self.maxs[feature]=max(values)
                self.mins[feature]=min(values)
            
        
    
    def transform(self, rows):
        unprocessedTexts=rows['fall_description']
        clusters=[]
        for tr in unprocessedTexts:
            # you can do more with the training text here and generate more features...
            clusters.append(getclusterfeatures(tr))
        
        data_vectors = self.clustervectorizer.transform(clusters).toarray()
        
        allcnums=rows['CNums']
        cnum_cluster_vectors = self.cnumclustervectorizer.transform(allcnums).toarray()
        allloccnums=rows['LocCNums']
        loc_cnum_cluster_vectors = self.loccnumClusterVectorizer.transform(allloccnums).toarray()
        
        data_vectors = np.concatenate((data_vectors, cnum_cluster_vectors), axis=1)
        data_vectors = np.concatenate((data_vectors, loc_cnum_cluster_vectors), axis=1)

        
        #tack on all the other numeric features
        for feature in ['duration',]:
            values=rows[feature]
            normValues = np.array([getNormalizedList(values, self.maxs[feature], self.mins[feature])])
            data_vectors=np.concatenate((data_vectors, normValues.T), axis=1)
        
        return data_vectors
    
    def fit_transform(self, rows, y=None):
        self.fit(rows)
        return self.transform(rows)


# In[ ]:



clf=KNeighborsClassifier(n_neighbors=1)

abDF=pd.DataFrame()

for vectorizer in [ablation1(), ablation2(), ablation3(), ablation4(), ablation5()]:

    #SIMPLE PIPELINE
    pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])

    grid_params = {}
    #SEARCH HYPERPARAMETERS
    folds = 5
    grid = grid_search_hyperparam_space(grid_params,pipeline,folds, training_rows,training_classes)

    #CLASSIFY AND EVALUATE 
    predictions_test = grid.predict(test_rows)
    
    macro=f1(predictions_test,test_classes, average='macro')
    micro=f1(predictions_test,test_classes, average='micro')
    print ("F1 Macro\t", macro)
    print ("F1 Micro\t", micro)
    print("\t****************************************\n")
    
    entry={"F1 Macro":macro, "F1 Micro":micro}
    abDF=abDF.append(entry, ignore_index=True)
    
abDF['Features Removed']=['Fall Duration', 'Metamap Location Tags', 'Metamap Description Tags', 
                          'TweetNLP Description Tags', 'N-Grams']
abDF


# ## Training size vs performance (F1 macro)

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=1)

x=[]
y=[]

for frac in np.arange(.4, 1.01, .02):
    
    partial_training_set_size=int(frac*training_set_size)
    partial_training_rows = training_rows.sample(n=partial_training_set_size)
    partial_training_classes=partial_training_rows['target'].tolist()
    
    vectorizer = myVectorizer()

    #SIMPLE PIPELINE
    pipeline = Pipeline(steps = [('vec',vectorizer),('classifier',clf)])

    grid_params = {}
    #SEARCH HYPERPARAMETERS
    folds = 5
    grid = grid_search_hyperparam_space(grid_params,pipeline,folds, partial_training_rows,partial_training_classes)

    #CLASSIFY AND EVALUATE 
    predictions_test = grid.predict(test_rows)
    
    x.append(partial_training_set_size)
    y.append(f1(predictions_test,test_classes, average='macro'))


# In[ ]:


from matplotlib import pyplot as plt

plt.plot(x,y)
plt.title("Performance vs Training Set Size")
plt.ylabel("F1 Macro Score")
plt.xlabel("Training Set Size")
plt.show()


# In[ ]:


#from https://www.kite.com/python/answers/how-to-plot-a-linear-regression-line-on-a-scatter-plot-in-python

plt.plot(x, y, 'o')

m, b = np.polyfit(x, y, 1)
plt.title("Performance vs Training Set Size")
plt.ylabel("F1 Macro Score")
plt.xlabel("Training Set Size")
plt.plot(x, [m*xi + b for xi in x])
plt.show()


# In[ ]:




