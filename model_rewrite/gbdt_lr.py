'''
@Author: your name
@Date: 2020-04-22 14:06:22
@LastEditTime: 2020-04-22 14:08:16
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/model_rewrite.py/gbdt_lr.py
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_root = '/Users/liulingzhi5/dataset/movielens/ml-1m/'
movies_dir = data_root + 'movies.dat'
users_dir = data_root + 'users.dat'
ratings_dir = data_root + 'ratings.dat'

movies = pd.read_csv(movies_dir,sep='::',header=None)
users = pd.read_csv(users_dir,sep='::',header=None)
ratings = pd.read_csv(ratings_dir,sep='::',header=None)
movies.columns=['MovieID', 'Title', 'Genres']
users.columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
ratings.columns=['UserID', 'MovieID', 'Rating', 'Timestamp']

data = ratings.merge(users, on=['UserID'])
data = data.merge(movies, on=['MovieID'])
data['hour'] = pd.to_datetime(data.UserID)
data['MovieID'] = LabelEncoder().fit_transform(data.MovieID)
