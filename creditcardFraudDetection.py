# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:21:45 2023

@author: dhair
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

df_credit_card = pd.read_csv('/Users/dhair/OneDrive/Desktop/creditcard.csv')
print(df_credit_card)

print(df_credit_card.describe())
print(df_credit_card.info())
#*************** Creating the New DataSet **************

fraud = df_credit_card[df_credit_card['Class'] == 1 ]
print('Fraud Detection : ' , fraud)

Non_fraud = df_credit_card[df_credit_card['Class'] == 0 ]
print('Not fraud Transaction : ' , Non_fraud)

#************** Calculate the Percentage *********

print('Precentage of fraud detection : ' , (len(fraud)/len(df_credit_card) ) *100, '%')

print('Percentage of verified transactions : ' , (len(Non_fraud)/len(df_credit_card) ) *100 , '%')

sns.countplot(df_credit_card['Class'] , label = "Count")

#*********** Find correlation ************
 
plt.figure(figsize = (30,10))
sns.heatmap(df_credit_card.corr() , annot = True)
plt.show()

column_headers = df_credit_card.columns.values
print(column_headers)

#***************** kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable *****************

i = 1

fig, ax = plt.subplots(8,4,figsize=(18,30))
for column_header in column_headers:    
    plt.subplot(8,4,i)
    sns.kdeplot(fraud[column_header], bw = 0.4, label = "Fraud", shade=True, color="r", linestyle="--")
    sns.kdeplot(Non_fraud[column_header], bw = 0.4, label = "Non Fraud", shade=True, color= "y", linestyle=":")
    plt.title(column_header, fontsize=12)
    i = i + 1
    plt.show()

#************* Create Training and Testing and Cleaning DateSet **********

sc = StandardScaler()
df_credit_card['Amount_Norm'] = sc.fit_transform(df_credit_card['Amount'].values.reshape(-1,1))
print(df_credit_card)

#*************** Drop the column ***************

df_credit_card = df_credit_card.drop(['Amount'], axis = 1)

#***************** Creating the X and Y Dataset *************
# Let's drop the target label coloumns
X = df_credit_card.drop(['Class'],axis=1)
y = df_credit_card['Class']
print(X)
print(y)

#**************** Train Test Split ************

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#*************** Apply Naive Bayes *************
 
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

#************ Evaluating the Model , confusion matrix **************

#*********** Predicting the Training Set results **************

y_predict_train = NB_classifier.predict(X_train)
print(y_predict_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
plt.show()

#********** Predicting the Test set results **************

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

#**************Classification Report *************

print(classification_report(y_test, y_predict_test))

#*************** Improving the Model ***************

X = df_credit_card.drop(['Time','V8','V13','V15','V20','V22','V23','V24','V25','V26','V27','V28','Class'], axis = 1)
#print(X)

#******** Apply Train Test Split and Apply Naive Bayes ***************

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_predict = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_predict))
print("Number of fraud points in the testing dataset = ", sum(y_test))







