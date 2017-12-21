# Twitter-Sentiment-Analysis
Twitter Sentiment Analysis with Linear SVM Classifier with TF-IDF and word2vec features.

# Introduction

This project describes how I gradually developed my understanding of text sentiment classification b y implementing an SVM classifier and a Logistic Regressor and performing
some further experiments to seek approaches that also work. The main system that I tried to implement for the final project is the Multi-view Ensemble approach by NILCUSP
that participated in SemEval 2017. NILC-USP uses three classifiers which is t rained in a different feature space using bag-of-words model and word embeddings to represent
sentences and uses a Linear SVM and a Logistic Regressor as base classifiers. Instead of implementing the voting ensemble, I focused more on how each of three classifiers was
built, strengths and drawbacks of the system, and how well it performs.

# Results

Test Results
AvgRec Accuracy
SVM /w Tf-idf 0.579 0.581
SVM /w Tf-idf and Curse Word 0.579 0.582
SVM /w Word Embeddings 0.548 0.600
LogisticReg /w Weighted Word2Vec 0.577 0.528
SVM /w Weighted Word2Vec 0.576 0.520
