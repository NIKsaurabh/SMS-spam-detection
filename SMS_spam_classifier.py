#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:10:03 2019

@author: saurabh
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


dataset = pd.read_csv('spam.csv',encoding='latin-1')
dataset.drop(columns={'Unnamed: 2','Unnamed: 3','Unnamed: 4'}, axis=1,inplace = True)
dataset.columns=['label', 'message']

lemmatize = WordNetLemmatizer()

for i in range(len(dataset)):
    dataset['message'][i] = re.sub('[^a-zA-Z]',' ',dataset['message'][i])
    dataset['message'][i] = dataset['message'][i].lower()
    word = nltk.word_tokenize(dataset['message'][i])
    word = [lemmatize.lemmatize(j) for j in word if j not in set(stopwords.words('english'))]
    dataset['message'][i] = ' '.join(word)
    
#creating bag of word using TfidfVectorizer
tv = TfidfVectorizer(max_features=2500)
X = tv.fit_transform(dataset['message']).toarray()

y=pd.get_dummies(dataset['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detection = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detection.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = spam_detection, X = X_train, y = y_train, cv=10)
acc.mean()
