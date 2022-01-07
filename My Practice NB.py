import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer # CountVectorizer is bag of words.
# features are nothing but keywords or tokens.
#TfidfTransformer - Tfidf weightage

# Loading the data set
email_data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Naive Bayes - Classifier Technique/sms_raw_NB.csv",encoding = "ISO-8859-1")
# for this particular data set uft not works so, "ISO-8859-1" works - for reading the data

# cleaning data 
import re # regular expression - it is used for string manipulation
stop_words = []
# Load the custom built Stopwords
with open("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Naive Bayes - Classifier Technique/stopwords_en.txt","r") as sw:
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

email_data.text = email_data.text.apply(cleaning_text)

# removing empty rows
email_data = email_data.loc[email_data.text != " ",:] # consider only non empty rows

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split # ramdomly split into train and test data

email_train, email_test = train_test_split(email_data, test_size = 0.2) # test file is 20%

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text) # How many times word is present in all the documents

# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text) 

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape # (row, column) , 6661 Keywords we extracted out of all data

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.type)
accuracy_test_m

#either apply this or this for accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, email_test.type) 

pd.crosstab(test_pred_m, email_test.type)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == email_train.type)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, email_train.type)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == email_test.type)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, email_test.type) 

pd.crosstab(test_pred_lap, email_test.type)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == email_train.type)
accuracy_train_lap


