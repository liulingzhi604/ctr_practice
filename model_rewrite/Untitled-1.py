'''
@Author: your name
@Date: 2020-04-26 15:51:51
@LastEditTime: 2020-04-26 15:51:51
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /code learn/ctr_practice/model_rewrite.py/Untitled-1.py
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
@Author: your name
@Date: 2020-04-22 10:09:07
@LastEditTime: 2020-04-22 10:33:16
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/model_rewrite.py/cf.py
'''

from sklearn.metrics import accuracy_score
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

# data = ratings.merge(users, on=['UserID'])
# data = data.merge(movies, on=['MovieID'])


# %%
# movies数据处理
movies['publish_date'] = movies.Title.str[-5:-1].astype(int)
movies['Title'] = LabelEncoder().fit_transform(movies.Title.str[:-7])

from sklearn.preprocessing import MultiLabelBinarizer
movie_genres = MultiLabelBinarizer().fit_transform(movies.Genres.map(lambda x: x.split('|')))
movie_genres = pd.DataFrame(movie_genres)
movie_genres.columns = ['Genres_%d' % i for i in range(len(movie_genres.columns))]
movies = pd.concat([movies, movie_genres], axis=1)
# users数据处理
users = users.drop(['Zip-code'], axis=1)
users['Gender'] = LabelEncoder().fit_transform(users.Gender)
# ratings数据处理
ratings = ratings.sort_values(['UserID', 'Timestamp'])
ratings['MovieID'] = ratings['MovieID'].astype(str)
watching_seq = ratings.groupby('UserID')['MovieID'].transform(lambda x: ','.join(x))
ratings['MovieID'] = ratings['MovieID'].astype(int)
ratings['watching_seq'] = ratings['UserID'].map(watching_seq)

dt = pd.to_datetime(ratings.Timestamp).dt
ratings['day'] = dt.day
ratings['hour'] = dt.hour
ratings = ratings.drop(['Timestamp'], axis=1)


# %%
data = ratings.merge(movies, on='MovieID')
data = data.merge(users, on='UserID')


# %%
data


# %%
data.dtypes


# %%
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, input_from_feature_columns, build_input_features, get_feature_names
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

sparse_features = ['UserID', 'MovieID', 'Title', 'Gender', 'Occupation']
dense_features = ['day', 'hour', 'publish_date', 'Age']
varlen_features = ['Genres']

genres_list = list(map(lambda x: x.split('|'), data['Genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)

# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)


# %%
data['publish_date'] = data['publish_date'].astype(int)


# %%
sparse_feature_columns = [SparseFeat(feat, data[feat].nunique() * 5,embedding_dim=4, use_hash=True, dtype='string')
                            for feat in sparse_features]
dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]                            
varlen_feature_columns = [VarLenSparseFeat(sparsefeat=SparseFeat('Genres', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"), maxlen=max_len,combiner= 'mean')]

# Notice : value 0 is for padding for sequence input feature

fixlen_feature_columns = sparse_feature_columns + dense_feature_columns

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns 
# dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
# linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

model_input = {name:data[name].values for name in feature_names}
# model_input['Genres'] = genres_list


# %%
for i in model_input:
    print(i,model_input[i].dtype)


# %%
model_input


# %%
data[['Rating']].values


# %%
model = DeepFM(linear_feature_columns,dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[['Rating']].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )


# %%


