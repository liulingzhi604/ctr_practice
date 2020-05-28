'''
@Author: your name
@Date: 2020-04-22 10:09:07
@LastEditTime: 2020-04-22 11:23:29
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/model_rewrite.py/cf.py
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
data = data.merge(movies, on=['MovieID'])[['UserID', 'MovieID', 'Rating']]
data['UserID'] = LabelEncoder().fit_transform(data.UserID)
data['MovieID'] = LabelEncoder().fit_transform(data.MovieID)


# 用于构建“评分矩阵” 然后 计算“相似矩阵”的数据
train_data = data[: int(0.8 * len(data))]
# 在已有的“相似矩阵”上进行测试的数据
test_data = data[int(0.8 * len(data)): ]

# 构建“评分矩阵”
rating_matrix = np.zeros(shape=[data.UserID.nunique(), data.MovieID.nunique()])
for idx in range(len(train_data)):
    sample = train_data.iloc[idx,:]
    rating_matrix[sample.UserID, sample.MovieID] = sample.Rating

# 计算行与行之间之间的cos距离，作为用户与用户之间的相似性
user_similarity = pairwise_distances(rating_matrix, metric='cosine')
# 计算列与列之间之间的cos距离，作为物品与物品之间的相似性
item_similarity = pairwise_distances(rating_matrix.T, metric='cosine')

# ※※※※※※※※※※到此为止， “相似矩阵”计算完毕，可以用于预测了， 预测只需要用到“相似矩阵”

def user_cf_pred(to_pred_rating_matrix, user_similarity):
    # 计算每一个用户的平均打分，因为不同的用户打分的基准有偏差，所以通过这个操作消除偏差
    mean_user_rating = to_pred_rating_matrix.mean(axis=1)
    adjusted_rating_matrix = to_pred_rating_matrix - mean_user_rating[:, np.newaxis]

    # 矩阵乘法  user_similarity · adjusted_rating_matrix
    #      [UserID, UserID]        ·     [UserID, ItemID]           =   [UserID, ItemID]
    # 一行：一个user对所有User的相似性 · 一列：所有user对一个item的打分     =   行列交叉的一个位置：一个user对一个item的打分，是由所有user对这个item的评分加权得到的
    pred = user_similarity.dot(adjusted_rating_matrix)/np.array([np.abs(user_similarity).sum(axis=1)]).T  + mean_user_rating[:, np.newaxis]
    return pred


def item_cf_pred(to_pred_rating_matrix, item_similarity):
    # 矩阵乘法 to_pred_rating_matrix · item_similarity
    #      [UserID, ItemID]        ·      [ItemID, ItemID]          =   [UserID, ItemID]
    # 一行：一个user对所有Item的评分   · 一列：一个item对所有item的相似性   =   行列交叉的一个位置：一个user对一个item的打分，是由这个user对所有item的评分加权得到的
    pred = to_pred_rating_matrix.dot(item_similarity)/np.array([np.abs(item_similarity).sum(axis=1)])
    return pred
