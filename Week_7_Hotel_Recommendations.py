'''
File              : Week_7_Hotel_Recommendations.py
Name              : Senthilraj Srirangan
Date              : 04/25/2020
Assignment Number : 7.3 Assignment: Create Optimal Hotel Recommendations
Course            : DSC 630 - Predictive Analytics
Exercise Details  :
                All online travel agencies are scrambling to meet the Artificial Intelligence driven personalization standard set by Amazon and Netflix.
                In addition, the world of online travel has become a highly competitive space where brands try to capture our attention (and wallet) with recommending,
                comparing, matching, and sharing. For this assignment, we aim to create the optimal hotel recommendations for Expedia’s users that are searching for a hotel to book.
                For this assignment, you need to predict which “hotel cluster” the user is likely to book, given his (or her) search details.
                In doing so, you should be able to demonstrate your ability to use four different algorithms (of your choice).
                The data set can be found at Kaggle: Expedia Hotel Recommendations To get you started,
                I would suggest you use train.csv which captured the logs of user behavior, and destinations.csv which contains information related to hotel reviews made by users.
                You are also required to write a one page summary of your approach in getting to your prediction methods. I expect you to use a combination of R and Python in your answer.
'''

import calendar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.translate import metrics
from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Process the train.csv data
train_data = pd.read_csv('C://MS/Course_7_630_Predictive_Analytics/Week_7/expedia-hotel-recommendations/train.csv', nrows=1000000)
train_data = train_data.dropna()

correlation = train_data.corr()['hotel_cluster'].sort_values()
print(correlation)

def fill_na_median(data, inplace=True):
    return data.fillna(data.median(), inplace=inplace)


fill_na_median(train_data['orig_destination_distance'])

# Feature Extraction,
train_data['date_time'] = pd.to_datetime(train_data['date_time'])
train_data['srch_ci'] = pd.to_datetime(train_data['srch_ci'])
train_data['srch_co'] = pd.to_datetime(train_data['srch_co'])

train_data['year'] = train_data['date_time'].dt.year
train_data['month'] = train_data['date_time'].dt.month

train_data['srch_ci_year'] = train_data['srch_ci'].dt.year
train_data['srch_ci_month'] = train_data['srch_ci'].dt.month

train_data['srch_co_year'] = train_data['srch_co'].dt.year
train_data['srch_co_month'] = train_data['srch_co'].dt.month

# 2. Process destination data
destination_data = pd.read_csv('C://MS/Course_7_630_Predictive_Analytics/Week_7/expedia-hotel-recommendations/destinations.csv')
destination = destination_data.dropna()

# Merge the Train and Destination Data

train_data = pd.merge(train_data, destination, how='left', on='srch_destination_id')
train_data.fillna(0, inplace=True)

# Filter out the non-booking rows, bring only the booked rows.

train_data = train_data.loc[train_data['is_booking'] == 1]

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['user_id', 'is_booking', 'hotel_cluster', 'date_time', 'srch_ci', 'srch_co'], axis=1)
train_y = train_data['hotel_cluster']   # Target Variable

# Train Test Split

#X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.50)

# 1. K Nearest Neighbour

model = KNeighborsClassifier(metric='euclidean')

# fit the model with the training data

model.fit(train_x, train_y)

# Number of Neighbors used to predict the target
print('\nThe number of neighbors used to predict the target : ', model.n_neighbors)


# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data', predict_train)

print('Acuracy Score : ',accuracy_score(train_y, predict_train))

# Random Forest

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.30)
print(len(X_train),len(X_test))

model = RandomForestClassifier()
y_pred = model.fit(X_train,y_train)

print('Random Forest Acuracy Score : ',model.score(X_test,y_test))

# Naive Bayes

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.30)
model = GaussianNB()
model.fit(X_train,y_train)
print('Naive Bayes Acuracy Score : ',model.score(X_test,y_test))

# Logistic Regression Tree
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
print('Logistic Acuracy Score : ',model.score(X_test,y_test))
