import gensim
import numpy as np

from gensim.models.keyedvectors import KeyedVectors

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import csv
import numpy as np
from sklearn.svm import LinearSVC

tweet_data_path = 'data/twitter-2016test-A.txt'
path = r'D:\GoogleNews-vectors-negative300.bin'
tweet_tokenizer = TweetTokenizer()
curse_word_path = 'data/google_badlist.txt'

# tweet_data = ['dear @Microsoft the newOoffice for Mac is great and all, but no Lync update? C\'mon.', 'If you haven\'t seen @iambigbirdmovie from my husband @chadnwalker, catch it on Amazon Prime starting Sept 5th! http://t.co/gjOyPozJZT']
tweet_data = []

with open(tweet_data_path , encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\t")
    tweet_data = list(reader)

parsed_tweet = []

# stop words

stop = set(stopwords.words('english'))

# get curse words

with open(curse_word_path, encoding = 'utf-8') as f:
    curse_words = [line.strip() for line in f]




for info in tweet_data:
    l = " ".join(tweet_tokenizer.tokenize(info[2].lower())).split(" ")
    filtered_sentence = [w for w in l if not w in stop and not w in string.punctuation
                         and ( w[0] != '@' and w[0] != '#' and w[:4] != 'http' )]
    #print(filtered_sentence)
    parsed_tweet.append(filtered_sentence)

curse_words = set(curse_words)

curse_vector = [len(curse_words.intersection(sent)) for sent in parsed_tweet]


# label the data

tweet_target = np.zeros(len(tweet_data))

for i in range(len(tweet_data)):
    if tweet_data[i][1] == 'negative':
        tweet_target[i] = 0
    elif tweet_data[i][1] == 'neutral':
        tweet_target[i] = 1
    elif tweet_data[i][1] == 'positive':
        tweet_target[i] = 2

model = KeyedVectors.load_word2vec_format(path, binary=True)

""" 
80% Training , 20% Testing
"""

twenty_percent = len(tweet_data) * 0.2

# Initialize word2vec_feature vector
total_svm = 0
word2vec_feature = []

# adds the word2vec average
for tweet in parsed_tweet:
    average_vec = np.zeros(300)
    for word in tweet:
        if word in model.wv:
            average_vec += (model.wv[word] / len(tweet))
        else:
            pass
    word2vec_feature.append(average_vec)

# concatenate curse vector
word2vec_feature.append(curse_vector)

tweet_tobe_trained = parsed_tweet[: -int(twenty_percent)]
tweet_tobe_teset = parsed_tweet[-int(twenty_percent):]

X_train = word2vec_feature[: -int(twenty_percent)]
y_train = tweet_target[: -int(twenty_percent)]

X_test = word2vec_feature[-int(twenty_percent):]
y_test = tweet_target[-int(twenty_percent):]

svc_model = LinearSVC()
svc_model.fit(X_train, y_train)

result1 = svc_model.predict(X_test)

total_svm = total_svm + sum(y_test == result1)

# Calculate Average Recall

fn_positive = 0
tp_positive = 0

for i, j in zip(y_test, result1):
    if i == 2 and i != j:
        fn_positive += 1
    if i == 2 and i == j:
        tp_positive += 1

fn_neutral = 0
tp_neutral = 0

for i, j in zip(y_test, result1):
    if (i == 1 and i != j):
        fn_neutral += 1
    if i == 1 and i == j:
        tp_neutral += 1

fn_negative = 0
tp_negative = 0

for i, j in zip(y_test, result1):
    if (i == 0 and i != j):
        fn_negative += 1
    if i == 0 and i == j:
        tp_negative += 1

recall_pos = tp_positive / (tp_positive + fn_positive)
recall_neg = tp_negative / (tp_negative + fn_negative)
recall_neu = tp_neutral / (tp_neutral + fn_neutral)

print('Average Recall : ', (1/3) * (recall_neg + recall_neu + recall_pos))
### Done Average Recall ###

print(total_svm/ (int(twenty_percent)) )
print(total_svm, ' out of ', (int(twenty_percent)))

