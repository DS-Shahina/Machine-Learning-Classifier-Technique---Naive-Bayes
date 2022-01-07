"Task 2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NB_Car_Ad= pd.read_csv('D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Naive Bayes - Classifier Technique/NB_Car_Ad.csv')

NB_Car_Ad.columns

X = NB_Car_Ad.iloc[:, [2, 3]].values
y = NB_Car_Ad.iloc[:, 4].values

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
"Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, 
"it is also known as data normalization and is generally performed during the data preprocessing step."

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting classifier to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()

train_pred_ber=classifier.fit(X_train,y_train).predict(X_train)
test_pred_ber=classifier.fit(X_train,y_train).predict(X_test)

train_acc_ber=np.mean(train_pred_ber==y_train)
test_acc_ber=np.mean(test_pred_ber==y_test)
train_acc_ber#0.795
test_acc_ber#0.794

pd.crosstab(test_pred_ber, y_test)

