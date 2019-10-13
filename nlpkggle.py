# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 06:23:27 2019

@author: Arham Jain
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

train.columns.values
print (train["review"][0])
print (train.head)

#removing html using beautiful soup
from bs4 import BeautifulSoup as bs
eg1=bs(train["review"][0])
print(eg1.get_text())
print (train["review"][0])

import re
eg1=re.sub('[^a-zA-Z]',' ',eg1.get_text())
eg1=eg1.lower()
eg1=eg1.split()

import  nltk
from nltk.corpus import stopwords
stopwrds=stopwords.words("english")
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
words= [ps.stem(w) for w in eg1 if not w in stopwrds]
print (words)
print((" ".join(words)))
print (review_to_words(train["review"][0]))


def review_to_words(raw_review):
    review=bs(raw_review)
    review=review.get_text()
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower().split()
    stopwrds=stopwords.words("english")
    ps=PorterStemmer()
    review= [ps.stem(w) for w in review if not w in stopwrds]
    return( " ".join( review ))  


num_reviews=train["review"].size

clean_review=[]
for i in range (0 ,num_reviews):
    clean_review.append(review_to_words(train["review"][i]))
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(clean_review).toarray()
Y=train.iloc[:,1].values
names=cv.get_feature_names()
print(names)
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100)
forest.fit(X,Y)    


test=pd.read_csv('testData.tsv',delimiter='\t',quoting=3)
clean_test=[]
num=test["review"].size
for i in range (0,num):
    clean_test.append(review_to_words(test["review"][i]))
test_features=cv.transform(clean_test).toarray()
out=forest.predict(test_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":out} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
