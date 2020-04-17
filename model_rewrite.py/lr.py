'''
@Author: your name
@Date: 2020-04-16 15:25:58
@LastEditTime: 2020-04-16 16:29:40
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ctr_practice/model_rewrite.py/lr.py
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
tf.__version__


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

X = data[dense_features + sparse_features].values
y = data[['label']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ================= 自写LR =================
sess = tf.Session()

x_inps = tf.placeholder(shape=[None, 39], dtype=tf.float32)
y_inps = tf.placeholder(shape=[None,1], dtype=tf.float32)

weight = tf.Variable(initial_value=tf.random.normal(shape=[39, 1]))
l2 = tf.nn.l2_loss(weight)
l1 = tf.reduce_sum(tf.abs(weight))
bias = tf.Variable(initial_value=np.zeros(shape=[1]), dtype=tf.float32)

logits = tf.matmul(x_inps, weight) + bias
prob = tf.sigmoid(logits)


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_inps, logits=logits) + l1

optimizer = tf.train.GradientDescentOptimizer(0.01)

step = optimizer.minimize(loss)


sess.run(tf.global_variables_initializer())
for i in range(1000):
    sess.run(
        fetches=[step],
        feed_dict={
            x_inps: X_train,
            y_inps: y_train
        })

test_y_pred = sess.run(
    fetches=[prob],
    feed_dict={
        x_inps: X_test,
        y_inps: y_test
    })
test_y_pred = list(np.array(test_y_pred).flatten())
test_y_pred = [round(i) for i in test_y_pred]

print(accuracy_score(y_test, test_y_pred))

lr = LogisticRegression()
lr.fit(X_train, y_train)
test_y_pred = lr.predict(X_test)
print(accuracy_score(y_test, test_y_pred))
