# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:37:12 2017

@author: YJ
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 03:55:04 2017

@author: YJ
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import csv
import numpy as np


from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

tweet_data_path = 'data/twitter-2016test-A.txt'
curse_word_path = 'data/google_badlist.txt'

tweet_tokenizer = TweetTokenizer()

# tweet_data = ['dear @Microsoft the newOoffice for Mac is great and all, but no Lync update? C\'mon.', 'If you haven\'t seen @iambigbirdmovie from my husband @chadnwalker, catch it on Amazon Prime starting Sept 5th! http://t.co/gjOyPozJZT']
tweet_data = []

with open(tweet_data_path , encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\t")
    tweet_data = list(reader)

parsed_tweet = []

# get curse words

with open(curse_word_path, encoding = 'utf-8') as f:
    curse_words = [line.strip() for line in f]

# stop words

stop = set(stopwords.words('english'))
curse_words = set(curse_words)

for info in tweet_data:
    l = " ".join(tweet_tokenizer.tokenize(info[2].lower())).split(" ")
    filtered_sentence = [w for w in l if not w in stop and not w in string.punctuation 
                         and ( w[0] != '@' and w[0] != '#' and w[:4] != 'http' )]
    
    #print(filtered_sentence)
    parsed_tweet.append(filtered_sentence)

curse_vector = [len(curse_words.intersection(sent)) for sent in parsed_tweet]

# creates a corpus with each document (tweet) having one string

for i in range(len(parsed_tweet)):
    parsed_tweet[i] = ' '.join(parsed_tweet[i])    

# label the data

tweet_target = np.zeros(len(tweet_data))

for i in range(len(tweet_data)):
    if tweet_data[i][1] == 'negative':
        tweet_target[i] = 0
    elif tweet_data[i][1] == 'neutral':
        tweet_target[i] = 1
    elif tweet_data[i][1] == 'positive':
        tweet_target[i] = 2
    

total_svm = 0

""" 
80% Training , 20% Testing
"""

twenty_percent = len(tweet_data) * 0.2

X_train = parsed_tweet[: -int(twenty_percent)]
y_train = tweet_target[: -int(twenty_percent)]

X_test = parsed_tweet[-int(twenty_percent):]
y_test = tweet_target[-int(twenty_percent):]

curse_train = curse_vector[: -int(twenty_percent)]
curse_test = curse_vector[-int(twenty_percent):]

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)

# Returns a feature vectors matrix having a fixed length tf-idf weighted word count feature
# for each document in training set. aka Term-document matrix

train_corpus_tf_idf = vectorizer.fit_transform(X_train)
test_corpus_tf_idf = vectorizer.transform(X_test)

curse_train = np.asarray(curse_train)
curse_test = np.asarray(curse_test)

train_corpus_tf_idf = np.concatenate((train_corpus_tf_idf.toarray(), curse_train.T[:, None]), axis = 1)
test_corpus_tf_idf = np.concatenate((test_corpus_tf_idf.toarray(), curse_test.T[:, None]), axis = 1)

model1 = LinearSVC()
model1.fit(train_corpus_tf_idf, y_train)

result1 = model1.predict(test_corpus_tf_idf)
total_svm = total_svm + sum(y_test == result1)

print(total_svm/ (int(twenty_percent)) )
print(total_svm, ' out of ', (int(twenty_percent)))

total_svm = 0


# initialize the K-cross fold validation so that the data-set is partitioned in 10 parts
# 1 part is used for testing and other 9 parts for training

kf = StratifiedKFold(n_splits=10)

for train_index, test_index in kf.split(parsed_tweet, tweet_target):
    X_train = [parsed_tweet[i] for i in train_index]
    X_test = [parsed_tweet[i] for i in test_index]
    y_train, y_test = tweet_target[train_index], tweet_target[test_index]
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
    train_corpus_tf_idf = vectorizer.fit_transform(X_train)
    test_corpus_tf_idf = vectorizer.transform(X_test)
    
    model1 = LinearSVC()
    model1.fit(train_corpus_tf_idf, y_train)
    result1 = model1.predict(test_corpus_tf_idf)
    
    total_svm = total_svm + sum(y_test == result1)
 
    
# Calculate Average Recall

fn_positive = 0
tp_positive = 0

for i,j in zip(y_test, result1):
    if i == 2 and i != j:
        fn_positive += 1
    if i == 2 and i == j:
        tp_positive += 1
        
fn_neutral = 0
tp_neutral = 0

for i,j in zip(y_test, result1):
    if(i == 1 and i != j):
        fn_neutral += 1
    if i == 1 and i == j:
        tp_neutral += 1
        
fn_negative = 0
tp_negative = 0

for i,j in zip(y_test, result1):
    if(i == 0 and i != j):
        fn_negative += 1
    if i == 0 and i == j:
        tp_negative += 1

recall_pos = tp_positive / (tp_positive + fn_positive)
recall_neg = tp_negative / (tp_negative + fn_negative)
recall_neu = tp_neutral / (tp_neutral + fn_neutral)

### Done Average Recall ###

print(total_svm/len(tweet_data))
print(total_svm, ' out of ', len(tweet_data))
print('Average Recall : ', (1/3) * (recall_neg + recall_neu + recall_pos))


"""
sklearn_tfidf = TfidfVectorizer(norm='l2', min_df = 0, use_idf = True, smooth_idf = False, sublinear_tf = True, tokenizer = tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(tweet_documents)

tf_idf_feature_data = []

for feature in sklearn_representation.toarray():
    tf_idf_feature_data.append(feature)
    

X = tf_idf_feature_data
y = tweet_target
C = 1.0 # SVM Regularization parameter

# SVC with linear kernel
svc = svm.SVC(kernel = 'linear', C = C).fit(X, y)

# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
"""