'''
@Author: your name
@Date: 2020-04-16 15:26:05
@LastEditTime: 2020-04-16 18:09:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/model_rewrite.py/fm.py
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict, namedtuple
from tensorflow.python.ops import init_ops
tf.__version__


# ================= 自写FM =================
SparseFeat = namedtuple('SparseFeat', ['name','embedding_dim', 'vocabulary'])
DenseFeat = namedtuple('DenseFeat', ['feature_num'])

class FM():
    def __init__(self, fixlen_feature_columns):
        self.build(fixlen_feature_columns)
        self.compile()


    def build(self, fixlen_feature_columns):
        self.input_dict = OrderedDict()
        self.emb_matrix_dict = OrderedDict()
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='y')


        # 二阶部分
        sparse_vec = []
        for feat in fixlen_feature_columns:
            if isinstance(feat, SparseFeat):
                self.input_dict[feat.name] = tf.placeholder(dtype=tf.int32, shape=[None, 1], name=feat.name)
                self.emb_matrix_dict[feat.name] = tf.Variable(initial_value=tf.random.normal(shape=[feat.vocabulary, feat.embedding_dim]))
                vec = tf.nn.embedding_lookup(self.emb_matrix_dict[feat.name], self.input_dict[feat.name])
                vec = tf.reshape(vec, shape=[-1, feat.embedding_dim])
                sparse_vec.append(vec)

        
        concat_vec =tf.concat(axis=1, values=sparse_vec)

        second_ord_logits = tf.layers.dense(inputs=concat_vec, units=1, activation=tf.nn.relu, kernel_initializer=init_ops.he_normal(), kernel_regularizer=tf.nn.l2_normalize)



        # 一阶部分
        for feat in fixlen_feature_columns:
            if isinstance(feat, DenseFeat):
                self.input_dict['dense'] = tf.placeholder(dtype=tf.float32, shape=[None, feat.feature_num], name='dense')

        first_ord_logits = tf.layers.dense(inputs=self.input_dict['dense'], units=1, activation=tf.nn.relu, kernel_initializer=init_ops.he_normal(), kernel_regularizer=tf.nn.l2_normalize)

        logits = first_ord_logits + second_ord_logits

        self.prob = tf.sigmoid(logits)

    def compile(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.prob)
        optimizer = tf.train.AdamOptimizer()
        self.step = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, feed_dict):
        self.sess.run(self.step, feed_dict)
        

    def predict(self, feed_dict):
        return self.sess.run(self.prob, feed_dict)



data_dir = '/Users/liulingzhi5/Desktop/code learn/DeepCTR/examples/criteo_sample.txt'
data = pd.read_csv(data_dir)
dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]
print(dense_features)
print(sparse_features)

data = data.fillna(0)
for feat in sparse_features:
    label_enc = LabelEncoder()
    data[feat] = data[feat].astype(str)
    data[feat] = label_enc.fit_transform(data[feat])

data = data.sample(frac=1)

X = data[dense_features + sparse_features]
y = data[['label']]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
X_train= X[:int(0.8*len(X))]
y_train= y[:int(0.8*len(X))]
X_test= X[int(0.8*len(X)):]
y_test= y[int(0.8*len(X)):]

sparse_feature_columns = [SparseFeat(name=feat, embedding_dim=4, vocabulary=data[feat].nunique()) for feat in sparse_features]
dense_feature_columns = [DenseFeat(feature_num=len(dense_features))]
fixlen_feature_columns = sparse_feature_columns + dense_feature_columns
features = sparse_features + dense_features



fm = FM(fixlen_feature_columns)
train_feed_dict = {fm.input_dict[feat.name] : X_train[[feat.name]] for feat in sparse_feature_columns}
train_feed_dict[fm.input_dict['dense']] =  X_train[dense_features]
train_feed_dict[fm.y] =  y_train

test_feed_dict = {fm.input_dict[feat.name] : X_test[[feat.name]] for feat in sparse_feature_columns}
test_feed_dict[fm.input_dict['dense']] =  X_test[dense_features]
test_feed_dict[fm.y] =  y_train

fm.fit(train_feed_dict)
test_y_pred = fm.predict(test_feed_dict)
test_y_pred = list(np.array(test_y_pred).flatten())
y_test = y_test.values
y_test = list(np.array(y_test).flatten())

nyp = []
for y in test_y_pred:
    if y > 0.8:
        nyp.append(1)
    else:
        nyp.append(0)

print(y_test)
print(nyp)
print(accuracy_score(y_test, nyp))

lr = LogisticRegression()
lr.fit(X_train, y_train)
test_y_pred = lr.predict(X_test)
print(accuracy_score(y_test, nyp))
