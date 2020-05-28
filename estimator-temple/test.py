'''
@Author: your name
@Date: 2020-05-27 18:16:27
@LastEditTime: 2020-05-28 13:52:19
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /estimator/test.py
'''
import pandas as pd

import tensorflow as tf

embedding_matrix = tf.constant(
    [
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
    ]
)
id = tf.constant([1])
res = tf.nn.embedding_lookup(embedding_matrix, id)
print(res.shape)
sess = tf.Session()

print(sess.run(res))