'''
@Author: your name
@Date: 2020-05-28 10:11:37
@LastEditTime: 2020-05-28 10:12:24
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /estimator/udf_model_fn.py
'''
import tensorflow as tf
def model_fn(features, labels, mode, params):
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" %
                        (name, features[name].shape))

    # 1. 如果要对embedding矩阵进行预训练赋值，可以在params中进行参数传递
    if 'embedding_initializer' in params:
        embedding_initializer = params['embedding_initializer']
    else:
        embedding_initializer = None

    # 2. 把input_fn传入给model，建立模型的graph，并返回需要调用的变量
    #  input_fn其实相当于一个已经在model外部就建立好的node，现在传给model，合并到graph中
    #  input_fn相当于一个placeholder node
    (total_loss, logits) = create_model(features)

    # 3. 在建立好graph之后就可以把checkpoint里面的变量值赋值给graph
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    # 4. 
    output_spec = None
    predictions = {"output": logits}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

            
    # 5. 定义模型在train  eval  test 三个部分，分别进行的计算
    # 这里定义的是output_spec
    # tf.estimator.train定义的是input_spec
    # input_spec配置的是
    #   输入
    #   训练多少步
    # output_spec配置的是
    #   loss
    #   训练步的操作
    #   summary的操作
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op, grad_summaries_op = create_optimizer(
            total_loss, learning_rate)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=FLAGS.output_dir,
            summary_op=grad_summaries_op)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            training_hooks=[summary_hook],
            export_outputs=export_outputs)

    elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(per_example_loss, label_ids, logits, is_real_example):
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.metrics.accuracy(
                labels=label_ids, predictions=predictions, weights=is_real_example)
            loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
            }

        eval_metrics = metric_fn(per_example_loss, label_id, logits, is_real_example)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics)
    else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities},
            export_outputs=export_outputs)
    return output_spec
