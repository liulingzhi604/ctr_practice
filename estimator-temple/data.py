'''
@Author: your name
@Date: 2020-05-27 14:54:07
@LastEditTime: 2020-05-28 15:39:49
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/estimator/convert2tfrecord.py
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict
import multiprocessing
import gc
ExampleStruct = None


def dataframe2tfrecord(output_file, examples, feature_spec, columns):
    global ExampleStruct
    ExampleStruct = namedtuple('ExampleStruct', columns)

    writer = tf.python_io.TFRecordWriter(output_file)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            tf.logging.info('Writing example %d of %d' % (idx, len(examples)))

        features = OrderedDict()
        for num_feat, feat in enumerate(columns):
            if feature_spec[feat] == 'str':
                features[feat] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(example[num_feat])]))
            elif feature_spec[feat] == 'int':
                features[feat] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(example[num_feat])]))
            elif feature_spec[feat] == 'float':
                features[feat] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(example[num_feat])]))
            else:
                raise ValueError('类型%s不可识别, 只接受 str  int  float' % df[feat].dtype)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()    


def csv2tfrecord(input_file, output_file, feature_spec, split=False):
    df = pd.read_csv(input_file)
    columns = list(df.columns)
    examples = df.values
    example_num = len(df)
    del df
    gc.collect()

    if split:
        dataframe2tfrecord(output_file+'.train', examples[: int(example_num * 0.8)], feature_spec, columns)
        dataframe2tfrecord(output_file+'.eval', examples[int(example_num * 0.8):], feature_spec, columns)
        return int(example_num * 0.8)
    dataframe2tfrecord(output_file, examples[: int(example_num * 0.8)], feature_spec, columns)
    return example_num


def tfrecord2fn(input_file, name2features, batch_size, num_epochs,  drop_remainder=True, is_training=True, target=None):
    if is_training and target is None:
        raise ValueError('训练阶段target不能为None')

    def _decode_record(record, name2features):
        example = tf.parse_single_example(record, name2features)

        for name in list(example.keys()):
            value = example[name]
            if value.dtype == tf.int64:
                value = tf.to_int32(value)
            example[name] = value
        return example

    def input_fn():
        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=100)


        dataset = tf.contrib.learn.read_batch_features(
            input_file, batch_size, name2features, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=max(4, multiprocessing.cpu_count()))

        # dataset的每一行是一个record， apply就是对每个record进行操作
        # dataset = dataset.map(
        #         map_func=lambda record: _decode_record(record, name2features),
        #     )
        # dataset = dataset.batch(
        #         batch_size=batch_size,
        #         drop_remainder=drop_remainder
        #     )

        label = None
        if is_training:
            label = dataset.pop(target)
        return dataset, label

    return input_fn
