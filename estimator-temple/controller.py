'''
@Author: your name
@Date: 2020-05-27 14:53:55
@LastEditTime: 2020-05-28 16:21:37
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /model-building/recommend/estimator/fm.py
'''
import tensorflow as tf
import os
from models.fm import FMConfig
from models.inputs import DenseFeature, SparseFeature
from model import model_fn_builder
from data import tfrecord2fn,csv2tfrecord
tf.logging.set_verbosity(tf.logging.INFO)
# =================================  预先写好tfrecord =================================
feature_spec = {
    'age':'int', 
    'sex':'int',
    'cp':'int',
    'trestbps':'int',
    'chol':'int',
    'fbs':'int',
    'restecg':'int',
    'thalach':'int',
    'exang':'int',
    'oldpeak':'int',
    'slope':'int',
    'ca':'int',
    'thal':'int',
    'target':'float'
}

split = True
input_file = '~/dataset/heart.csv'
output_file = './heart.tfrecord'
example_num = csv2tfrecord(input_file, output_file, feature_spec, split)


# =================================  环境配置 =================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 128
num_epochs = 10000
num_train_steps = example_num / batch_size * num_epochs


# =================================  模型定义 =================================
dense_features = ['age','trestbps','chol','thalach']
sparse_features = ['sex','cp','fbs','restecg','exang','oldpeak','slope','ca','thal']
vocab_dict = {
    'sex':2,
    'cp':4,
    'fbs':2,
    'restecg':3,
    'exang':2,
    'oldpeak':40,
    'slope':3,
    'ca':5,
    'thal':4
}

sparse_feature_columns = [SparseFeature(feature_name=feat, vocab_size=vocab_dict[feat], embedding_dim=16) for feat in sparse_features]
dense_feature_columns = [DenseFeature(feature_name=feat) for feat in dense_features]

model_config = FMConfig(sparse_feature_columns, dense_feature_columns, class_num=2)
model_fn = model_fn_builder(
        model_config=model_config, 
        learning_rate=0.001,
        init_checkpoint=None,
        summary_save_dir='./log/summary/', 
        summary_every_n_step=100,
        task='binary_classification'    
)


# =================================  estimator配置 =================================
session_config = tf.ConfigProto(allow_soft_placement=True)
run_config = tf.estimator.RunConfig(
    log_step_count_steps=1000,
    save_checkpoints_steps=1000,
    session_config=session_config,
    model_dir='./log/model'
)
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='./log/model',
    params={},
    config=run_config
)

# =================================  estimator执行 =================================
# ========================  构建输入 ========================
# 配置tfrecord的数据结构格式
name2features = {
    "age": tf.FixedLenFeature([], tf.int64),
    "sex": tf.FixedLenFeature([], tf.int64),
    "cp": tf.FixedLenFeature([], tf.int64),
    "trestbps": tf.FixedLenFeature([], tf.int64),
    "chol": tf.FixedLenFeature([], tf.int64),
    "fbs": tf.FixedLenFeature([], tf.int64),    
    "restecg": tf.FixedLenFeature([], tf.int64),    
    "thalach": tf.FixedLenFeature([], tf.int64),    
    "exang": tf.FixedLenFeature([], tf.int64),    
    "oldpeak": tf.FixedLenFeature([], tf.int64),    
    "slope": tf.FixedLenFeature([], tf.int64),    
    "ca": tf.FixedLenFeature([], tf.int64),    
    "thal": tf.FixedLenFeature([], tf.int64),    
    "target": tf.FixedLenFeature([], tf.float32),    
}
													
if split:
    train_input_fn = tfrecord2fn('heart.tfrecord.train', name2features, batch_size, num_epochs,drop_remainder=True, is_training=True, target='target')
    eval_input_fn = tfrecord2fn('heart.tfrecord.eval', name2features, batch_size, num_epochs, drop_remainder=True, is_training=True, target='target')

else:
    train_input_fn = tfrecord2fn('heart.tfrecord', name2features, batch_size, num_epochs,drop_remainder=True, is_training=True, target='target')
    eval_input_fn = tfrecord2fn('heart.tfrecord', name2features, batch_size, num_epochs, drop_remainder=True, is_training=True, target='target')
    test_input_fn = tfrecord2fn('heart.tfrecord', name2features, batch_size, num_epochs, drop_remainder=True, is_training=False)

# ========================  进行训练 ========================
early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
    estimator=estimator,
    metric_name='eval_loss',
    max_steps_without_decrease=1000,
    min_steps=10,
    run_every_secs=None,
    run_every_steps=1000
)

# estimator.train(
#     train_input_fn, max_steps=num_train_steps

# )
# print(estimator.evaluate(eval_input_fn))
# res = estimator.predict(test_input_fn)
# for  r in res:
#     print(r)
tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(train_input_fn, max_steps=num_train_steps, hooks=[early_stopping_hook]),
    eval_spec=tf.estimator.EvalSpec(eval_input_fn)
)

