"Task 1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salary_train=pd.read_csv('D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Naive Bayes - Classifier Technique/SalaryData_Train.csv')
salary_test=pd.read_csv('D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Naive Bayes - Classifier Technique/SalaryData_Test.csv')
salary_train.columns
salary_test.columns
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]

######### Naive Bayes ##############

#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
Gmodel=GaussianNB()
train_pred_gau=Gmodel.fit(train_X,train_Y).predict(train_X)
test_pred_gau=Gmodel.fit(train_X,train_Y).predict(test_x)

train_acc_gau=np.mean(train_pred_gau==train_Y)
test_acc_gau=np.mean(test_pred_gau==test_y)
train_acc_gau#0.795
test_acc_gau#0.794

pd.crosstab(test_pred_gau, test_y)

#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
Mmodel=MultinomialNB()
train_pred_multi=Mmodel.fit(train_X,train_Y).predict(train_X)
test_pred_multi=Mmodel.fit(train_X,train_Y).predict(test_x)

train_acc_multi=np.mean(train_pred_multi==train_Y)
test_acc_multi=np.mean(test_pred_multi==test_y)
train_acc_multi#0.772
test_acc_multi#0.774


pd.crosstab(test_pred_multi, test_y)
