
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:32:14 2019

@author: olevi
"""

#%%
import pandas as pd
import numpy as np
import codecs
import nltk
import pickle
import my_utils
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer,HashingVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,log_loss,precision_score,recall_score
from sklearn.metrics.pairwise import linear_kernel

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


from scipy.stats import pearsonr

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_files = my_utils.read_fake_satire_data("FakeNewsData/StoryText 2/")
headlines_data = data_files[0]
text_data = data_files[1]
target = data_files[2]

#%%
# SET TOKENIZER WITH STEMMING
# from nltk.stem.porter import *
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

# pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
pattern = r'\w+|\?|\!|\"|\'|\;|\:'


class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(pattern)
        self.stemmer = stemmer

    def __call__(self, doc):
        return [self.stemmer.stem(token) 
                for token in self.tok.tokenize(doc)]

#%%  
# SET PIPELINE WITH TOKENIZER, N-GRAMS AND LOGISTIC REGRESSION MODEL
clf_text = Pipeline([('vect', TfidfVectorizer()),
                      #('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
             

clf_text1 = Pipeline([('vect', TfidfVectorizer(tokenizer=Tokenizer(), stop_words='english', ngram_range=(1, 2), max_features=500)),
                      #('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
             
#%%
# SAVE CLASSIFIER
filename = 'text_classifier.pk'
with open(''+filename, 'wb') as file:
	pickle.dump(clf_text, file)


#%%
# LOAD CLASSIFIER
with open("text_classifier.pk" ,'rb') as f:
    clf_text= pickle.load(f)

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_text1, text_data, target, cv=10)  #scoring='roc_auc'
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
