"Task 3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer # CountVectorizer is bag of words.

Disaster_tweets_NB = pd.read_csv('D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Naive Bayes - Classifier Technique/Disaster_tweets_NB.csv')

Disaster_tweets_NB.columns

# cleaning data 
import re # regular expression - it is used for string manipulation
stop_words = []
# Load the custom built Stopwords
with open("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Naive Bayes - Classifier Technique/stopwords_en.txt","r") as sw:
    stop_words = sw.read() # because it is unstructure data i am not using pandas i am using open function

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower() # convert everthing to space and lower case
    i = re.sub("[0-9" "]+"," ",i) # convert everthing to space
    w = [] # empty list - is capture the keyword that we choose
    for word in i.split(" "): # i is each message and split, get tokens.
        if len(word)>3: # considering only those words which has greater than 3 letters, she, has kind of words are gone
            w.append(word) # if it's greater than 3 then append.
    return (" ".join(w)) # # Joinining all the msgs into single paragraph 

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")
cleaning_text("Hi how are you, I am good")

Disaster_tweets_NB.text = Disaster_tweets_NB.text.apply(cleaning_text)

# removing empty rows
Disaster_tweets_NB = Disaster_tweets_NB.loc[Disaster_tweets_NB.text != " ",:] # consider only non empty rows

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split # ramdomly split into train and test data

Disaster_tweets_NB_train, Disaster_tweets_NB_test = train_test_split(Disaster_tweets_NB, test_size = 0.2) # test file is 20%

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
Disaster_tweets_NB_bow = CountVectorizer(analyzer = split_into_words).fit(Disaster_tweets_NB.text) # How many times word is present in all the documents

# Defining BOW for all messages
all_Disaster_tweets_NB_matrix = Disaster_tweets_NB_bow.transform(Disaster_tweets_NB.text) 

# For training messages
train_Disaster_tweets_NB_matrix = Disaster_tweets_NB_bow.transform(Disaster_tweets_NB_train.text)

# For testing messages
test_Disaster_tweets_NB_matrix = Disaster_tweets_NB_bow.transform(Disaster_tweets_NB_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_Disaster_tweets_NB_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_Disaster_tweets_NB_matrix)
train_tfidf.shape # (row, column) , 6661 Keywords we extracted out of all data

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_Disaster_tweets_NB_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, Disaster_tweets_NB_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == Disaster_tweets_NB_test.target)
accuracy_test_m

#either apply this or this for accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, Disaster_tweets_NB_test.target) 

pd.crosstab(test_pred_m, Disaster_tweets_NB_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == Disaster_tweets_NB_train.target)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, Disaster_tweets_NB_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == Disaster_tweets_NB_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, Disaster_tweets_NB_test.target) 

pd.crosstab(test_pred_lap, Disaster_tweets_NB_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == Disaster_tweets_NB_train.target)
accuracy_train_lap


