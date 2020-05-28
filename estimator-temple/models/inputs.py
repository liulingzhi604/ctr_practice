'''
@Author: your name
@Date: 2020-05-27 13:33:49
@LastEditTime: 2020-05-28 14:17:18
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/fm/inputs.py
'''

import tensorflow as tf
from collections import namedtuple

SparseFeature = namedtuple('SparseFeature', ['feature_name', 'vocab_size', 'embedding_dim'])
DenseFeature = namedtuple('DenseFeature', ['feature_name'])


def build_input_placeholder(feature_columns):
    input_placeholders_dict = {}
    for feature in feature_columns:
        if isinstance(feature, SparseFeature):
            h = tf.placeholder(dtype=tf.float32, shape=[None, feature.embedding_dim])
        elif isinstance(feature, DenseFeature):
            h = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        else:
            raise ValueError('unknown')
        input_placeholders_dict[feature.feature_name] = h
    return input_placeholders_dict


def build_embedding_matrix_dict(sparse_feature_columns):
    embedding_matrix_dict = {}
    for feature in sparse_feature_columns:
        if not isinstance(feature, SparseFeature):
            raise ValueError('只有sparse_feature才能建立embedding矩阵')
        embedding_matrix = tf.get_variable(
            name=feature.feature_name+'_embedding_matrix',
            shape=[feature.vocab_size,feature.embedding_dim],
            initializer=tf.random_normal_initializer(),
            dtype=tf.float32
        )
        embedding_matrix_dict[feature.feature_name] = embedding_matrix
    return embedding_matrix_dict



# test = SparseFeature(embedding_dim=4)
# print(test.embedding_dim)