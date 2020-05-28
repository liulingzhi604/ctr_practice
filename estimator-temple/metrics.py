'''
@Author: your name
@Date: 2020-05-28 14:09:22
@LastEditTime: 2020-05-28 15:16:14
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /estimator/metrics.py
'''
import tensorflow as tf
def multicls_metric_fn(total_loss, labels, logits):
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions)
    loss = tf.metrics.mean(values=total_loss)
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss,
    }

def bicls_metric_fn(total_loss, labels, logits):
    predictions = tf.round(tf.nn.sigmoid(logits))
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions)
    loss = tf.metrics.mean(values=total_loss)
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss,
    }



def reg_metric_fn(total_loss, labels, logits):
    # 如果是回归的话logits就是一个数
    predictions = logits
    mse = tf.metrics.mean_squared_error(
        labels=labels, predictions=predictions)
    loss = tf.metrics.mean(values=total_loss)
    return {
        "eval_mse": mse,
        "eval_loss": loss,
    }
