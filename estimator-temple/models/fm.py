'''
@Author: your name
@Date: 2020-05-27 10:52:57
@LastEditTime: 2020-05-28 16:19:01
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/fm/fm.py
'''
import pandas as pd
import tensorflow as tf
from models.inputs import SparseFeature, DenseFeature, build_input_placeholder, build_embedding_matrix_dict
import six
import copy


class FMConfig():
    def __init__(self, sparse_feature_columns, dense_feature_columns, class_num):
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.class_num = class_num
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = FMConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class FM():
    # 当前理解(2020年05月27日 星期三)：
    # FM的基础还是embedding，公式推导中，即xixj<vi·vj>，把特征xi和特征xj映射为了vi,vj，在不同的特征向量之间做交互
    # 所有特征之间做的交互，实际上是一个方阵的下三角区域（不包含对角线， 对角线是自己和自己的交互，不再fm的计算范围之内），整个矩阵是关于对角线重复的，所以为了方便计算，算了(整个矩阵 - 额外一条对角线)/2
    #   整个矩阵：两个取和函数可以进行拆分，变成两个求和函数相乘，即所有vi，先取和，再平方（这里的平方其实是两个向量点积，每个元素平方后还需要加和）。
    #   对角线：向量自己的平方（这里的平方其实是两个向量点积，每个元素平方后还需要加和）在求和
    # 所以结果为 tf.squrae(sum(embedding_list)) - sum(tf.square(embedding_list))
    #  embedding_list 为

    #  batchsize *[
    #              [v11,v12,v13],
    #              [v21,v22,v23]
    #             ]
    #  sum希望的结果是 batchsize * [v11+v21, v12+v22, v13+v33], 所以是reduce_sum(axis=1),把1轴加掉
    # 希望的使用方式
    # tf.reduce_sum(tf.square(tf.reduce_sum(embedding_list, axis=1))) - tf.reduce_sum(tf.reduce_sum(tf.square(embedding_list), axis=1))
    # 最外层的tf.reduce_sum可以合并
    # tf.reduce_sum(tf.square(tf.reduce_sum(embedding_list, axis=1)) - tf.reduce_sum(tf.square(embedding_list), axis=1))
    # 
    def __init__(self, model_config, inputs, labels, scope='FM', mode=tf.estimator.ModeKeys.TRAIN):
        self.config = copy.deepcopy(model_config)
        self.inputs = inputs
        self.labels = labels

        with tf.variable_scope(scope, default_name='embeddings'):
            self.embedding_matrix_dict = build_embedding_matrix_dict(self.config.sparse_feature_columns)


        if len(self.config.dense_feature_columns) == 0:
            linear_logits = self.linear(self.config.dense_feature_columns)
            self.logits = linear_logits
        elif len(self.config.sparse_feature_columns) == 0:
            fm_logits = self.fm(self.config.sparse_feature_columns)
            self.logits = fm_logits
        else:
            linear_logits = self.linear(self.config.dense_feature_columns)
            fm_logits = self.fm(self.config.sparse_feature_columns)
            self.logits = linear_logits + fm_logits
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.loss = None
        else:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=labels))



    def linear(self, dense_feature_columns):
        dense_feature_names = [f.feature_name for f in dense_feature_columns]
        # print([self.inputs[f] for f in dense_feature_names])
        linear_input = tf.stack([self.inputs[f] for f in dense_feature_names], axis=1)
        linear_input = tf.cast(linear_input, tf.float32)

        linear_logits = tf.layers.dense(linear_input, 1)
        linear_logits = tf.reduce_sum(linear_logits, axis=1)
        return linear_logits

    def fm(self, sparse_feature_columns):
        sparse_feature_names = [f.feature_name for f in sparse_feature_columns]
        # 由于tfrecord里面的变量全都是list，所以这里的embedding lookup的结果维度应该是[bs, 1, dim]  错误
        # 可能并不是list， 
        fm_input = tf.stack(
            [tf.nn.embedding_lookup(self.embedding_matrix_dict[f], self.inputs[f]) for f in sparse_feature_names],
            axis=1
            )

        square_of_sum = tf.square(tf.reduce_sum(fm_input, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(fm_input), axis=1)
        fm_logits = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1)
        return fm_logits
    
            

    def get_logits(self):
        return self.logits            

    def get_loss(self):
        return self.loss


feed_dict = {}


# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="../data/")
# parser.add_argument("--output_path", type=str, default="./output/")
# parser.add_argument("--sparse_features", type=str, default="virginica")
# parser.add_argument("--embedding_dim", type=int, default=4)
# parser.add_argument("--target", type=str, default="virginica")
# parser.add_argument("--exclude", type=str, default="#")
# parser.add_argument("--n_estimators", type=int, default=100)
# args = parser.parse_args()


# data = pd.read_csv(args.data_dir)

# sparse_features = [feat.strip() for feat in args.sparse_features.split(',')]
# dense_features = [feat for feat in data.columns if feat not in sparse_features and feat not != args.target]
# target = args.target

# sparse_feature_columns = [SparseFeature(feature_name=feat, embedding_dim=args.embedding_dim) for feat in sparse_features]
# dense_feature_columns = [DenseFeature(feature_name=feat) for feat in dense_features]
# feature_columns = sparse_feature_columns + dense_feature_columns
# input_placeholders = build_input_placeholder(feature_columns)

# feed_dict = {feature:data[feature] for feature in input_placeholders}
# # 为什么要把placeholders放到类的外面来声明：因为需要用类名构建feed_dict
# #   其实为了保持类的封闭性，是可以把placeholder的构建放到类的里面的，并暴露一个get_placeholders的接口，返回类中构建的placeholder，构建feed_dict

# fm = FM(sparse_feature_columns, dense_feature_columns, input_placeholders)
# data = pd.DataFrame()
# fm.fit(feed_dict, data[target])
# fm.transform(data)
