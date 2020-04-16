'''
@Author: your name
@Date: 2020-04-16 15:13:30
@LastEditTime: 2020-04-16 15:20:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/system_build/offline/recall/fm.py
'''
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder

data_dir = '../dataset'
movies_dir = data_dir + '/movies.dat'
users_dir = data_dir + '/users.dat'
rating_dir = data_dir + '/ratings.dat'

movies = pd.read_table(movies_dir,sep='::',header=None)
users = pd.read_table(users_dir,sep='::',header=None)
rating = pd.read_table(rating_dir,sep='::',header=None)
movies.columns = ['MovieID', 'Title', 'Genres']
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
rating.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

movies['Year'] = movies.Title.str[-5:-1]
movies['Title'] = movies.Title.str[:-7]

users = users.drop('Zip-code', axis=1)

rating['day'] = pd.to_datetime(rating.Timestamp, unit='ms').dt.day
rating['weekday'] = pd.to_datetime(rating.Timestamp, unit='ms').dt.weekday
rating['hour'] = pd.to_datetime(rating.Timestamp, unit='ms').dt.hour
rating = rating.drop('Timestamp', axis=1)

data = rating.merge(movies, on='MovieID', how='left')
data = rating.merge(users, on='UserID', how='left')

data = data.sample(frac=1)

sparse_features = ['UserID', 'MovieID', 'Gender', 'Occupation', 'day', 'weekday']
dense_features = ['hour', 'Age']
for feat in sparse_features:
    label_enc = LabelEncoder()
    data[feat] = label_enc.fit_transform(data[feat])

feats = [i for i in data.columns if i != 'Rating']
X = data[feats]
y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sparse_features = ['UserID', 'MovieID', 'Gender', 'Occupation', 'day', 'weekday']
dense_features = ['hour', 'Age']

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for feat in sparse_features] + \
                         [DenseFeat(feat, 1) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='multiclass')
model.compile('adam', 'mse', metrics=['accuracy'])

feature_names = get_feature_names(fixlen_feature_columns)

train_feed_dict = {name: X_train[name] for name in feature_names}
test_feed_dict = {name: X_test[name] for name in feature_names}

model.fit(train_feed_dict, y_train, batch_size=256, epochs=10, validation_split=0.2)
pred_ans = model.predict(test_feed_dict, batch_size=256)