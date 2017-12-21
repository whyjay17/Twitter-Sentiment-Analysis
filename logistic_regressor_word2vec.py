import gensim
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import csv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

tweet_data_path = 'data/twitter-2016test-A.txt'
path = r'D:\GoogleNews-vectors-negative300.bin'
tweet_tokenizer = TweetTokenizer()

# tweet_data = ['dear @Microsoft the newOoffice for Mac is great and all, but no Lync update? C\'mon.', 'If you haven\'t seen @iambigbirdmovie from my husband @chadnwalker, catch it on Amazon Prime starting Sept 5th! http://t.co/gjOyPozJZT']
tweet_data = []

with open(tweet_data_path , encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\t")
    tweet_data = list(reader)

parsed_tweet = []

# stop words

stop = set(stopwords.words('english'))


for info in tweet_data:
    l = " ".join(tweet_tokenizer.tokenize(info[2].lower())).split(" ")
    filtered_sentence = [w for w in l if not w in stop and not w in string.punctuation
                         and ( w[0] != '@' and w[0] != '#' and w[:4] != 'http' )]
    #print(filtered_sentence)
    parsed_tweet.append(filtered_sentence)

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

model = KeyedVectors.load_word2vec_format(path, binary=True)

""" 
80% Training , 20% Testing
"""

twenty_percent = len(tweet_data) * 0.2

X_train = parsed_tweet[: -int(twenty_percent)]
y_train = tweet_target[: -int(twenty_percent)]

X_test = parsed_tweet[-int(twenty_percent):]
y_test = tweet_target[-int(twenty_percent):]

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)

# Returns a feature vectors matrix having a fixed length tf-idf weighted word count feature
# for each document in training set. aka Term-document matrix

train_corpus_tf_idf = vectorizer.fit_transform(X_train)
test_corpus_tf_idf = vectorizer.transform(X_test)

# Store the tf-idf of each word in a data structure

score_dict = defaultdict(lambda: defaultdict(lambda: float))
word_tfidf_vals = []
feature_names = vectorizer.get_feature_names()

# Get tfidf scores for training data
for index in range(len(X_train)):
    feature_index = train_corpus_tf_idf[index,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [train_corpus_tf_idf[index, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        score_dict[index][w] = s

# Get tfidf for test data
for index in range(len(X_test)):
    feature_index = test_corpus_tf_idf[index,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [test_corpus_tf_idf[index, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        score_dict[len(X_train) + index][w] = s

# Initialize word2vec_feature vector
total_svm = 0
word2vec_feature = []

# convert back to list of strings for word2vec usage

for i in range(len(parsed_tweet)):
    parsed_tweet[i] = parsed_tweet[i].split(' ')

# adds the word2vec average , multiply by the tfidf score of the word to the word2vec vector
for index in range(len(parsed_tweet)):
    average_vec = np.zeros(300)
    for word in parsed_tweet[index]:
        if word in model.wv:
            if word in score_dict[index]:
                weight = score_dict[index][word]
            else:
                weight = 1.0
            average_vec += ((model.wv[word] * weight) / len(parsed_tweet[index]))
        else:
            pass
    word2vec_feature.append(average_vec)




tweet_tobe_trained = parsed_tweet[: -int(twenty_percent)]
tweet_tobe_teset = parsed_tweet[-int(twenty_percent):]

X_train = word2vec_feature[: -int(twenty_percent)]
y_train = tweet_target[: -int(twenty_percent)]

X_test = word2vec_feature[-int(twenty_percent):]
y_test = tweet_target[-int(twenty_percent):]


# Logistic Regressor

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)

# SVM Model
#svc_model = LinearSVC()
#svc_model.fit(X_train, y_train)
#result1 = svc_model.predict(X_test)

result1 = logreg.predict(X_test)


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
