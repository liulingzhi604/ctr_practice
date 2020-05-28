'''
@Author: your name
@Date: 2020-05-27 16:46:45
@LastEditTime: 2020-05-28 16:06:02
@LastEditors: Please set LastEditors
@Description: In User Settings Edit)
@FilePath: /model-building/recommend/estimator/model.py
'''
import tensorflow as tf
import collections
from models.fm import FM, FMConfig
from metrics import bicls_metric_fn, multicls_metric_fn, reg_metric_fn
from models.inputs import DenseFeature, SparseFeature



def create_optimizer(loss, init_lr):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(init_lr)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # 梯度的计算是固定的，和优化器无关，优化器只是去自适应的确定学习率，所以操作流程是
    #     1. 梯度计算
    #     2. 梯度截断
    #     3. 优化器把梯度加到原变量中
    grad_summaries = []
    for g, v in zip(grads, tvars):
        if g is not None:
            grad_hist_summary = tf.summary.histogram(
                "{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar(
                "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", loss)
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    return (train_op, train_summary_op)


def udf_create_model(model_config, features, labels, embedding_initializer=None, mode=tf.estimator.ModeKeys.TRAIN):
    '''
    Parameters:
    -------------------
    model_config: 每个模型类对应一个config类，用于定义模型的各种参数
    features: 从input_fn解析出来的输入，作为一个tensor，类似placeholder
    embedding_initializer 如果有预训练的embdding要赋值的话从这里传入，默认为None
    '''
    

    model = FM(model_config=model_config, inputs=features, labels=labels, mode=mode)
    logits, loss = model.get_logits(), model.get_loss()

    return (logits, loss)


def model_fn_builder(
        model_config, learning_rate=None,
        init_checkpoint=None,
        summary_save_dir='./log', summary_every_n_step=100,
        task='binary_classification'):
    '''
    Parameters:
    -------------------
    model_config: 每个模型类对应一个config类，用于定义模型的各种参数
    init_checkpoint: checkpoint文件的路径。预训练embedding不再这里设置，在model_fn的param中传递
    learning_rate: 学习率
    summary_save_dir: 训练生成的checkpoint存储路径
    summary_every_n_step: 训练多少步存储一次ckpt
    task: cls 或者 reg, 对应不同的eval metrics


    model_fn一般情况不许要大幅度修改，但是在任务类型改变的时候，比如没有logits?其实回归也能当做logits只有1个
    '''
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))


        # ==================================================================
        # 1. 如果要对embedding矩阵进行预训练赋值，可以在params中进行参数传递
        if 'embedding_initializer' in params:
            embedding_initializer = params['embedding_initializer']
        else:
            embedding_initializer = None

        # ==================================================================
        # 2. 把input_fn传入给model，建立模型的graph，并返回需要调用的变量
        #  input_fn其实相当于一个已经在model外部就建立好的node，现在传给model，合并到graph中
        #  input_fn相当于一个placeholder node
        logits, total_loss = udf_create_model(
            model_config, features, labels, embedding_initializer, mode)
        # ==================================================================
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


        # ==================================================================
        # 4.
        output_spec = None
        predictions = {"output": logits}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}

        # ==================================================================
        # 5. 定义模型在train  eval  test 三个部分，分别进行的计算
        # 这里定义的是output_spec
        # tf.estimator.train定义的是input_spec
        # input_spec配置的是
        #   输入
        #   训练多少步
        #   hook的操作early_stop，训练多少步做一次eval

        # output_spec配置的是
        #   loss
        #   训练步的操作
        #   hook的操作summary，训练多少步记录一次tensorboard
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, grad_summaries_op = create_optimizer(
                total_loss, learning_rate)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=summary_every_n_step,
                output_dir=summary_save_dir,
                summary_op=grad_summaries_op)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[summary_hook],
                export_outputs=export_outputs)

        elif mode == tf.estimator.ModeKeys.EVAL:

            if task == 'multi_classification':
                eval_metrics = multicls_metric_fn(total_loss, labels, logits)
            elif task == 'binary_classification':
                eval_metrics = bicls_metric_fn(total_loss, labels, logits)
            elif task == 'regression':
                eval_metrics = reg_metric_fn(total_loss, labels, logits)
            else:
                raise ValueError('task 只能设置为[multi_classification, binary_classification, regression]')
            summaries_op = [tf.summary.scalar('eval_loss',total_loss)]
            for m in eval_metrics:
                summaries_op.append(tf.summary.scalar(m, eval_metrics[m][0]))
            summaries_op = tf.summary.merge(summaries_op)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=summary_every_n_step,
                output_dir=summary_save_dir,
                summary_op=summaries_op)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                evaluation_hooks=[summary_hook],
                eval_metric_ops=eval_metrics)
        else:

            if task == 'multi_classification':
                predictions = {
                    'probabilities': tf.nn.softmax(logits, axis=-1),
                    'class': tf.argmax(logits, axis=-1, output_type=tf.int32)
                }
            elif task == 'binary_classification':
                predictions = {
                    'probabilities': tf.nn.sigmoid(logits),
                    'class': tf.round(tf.nn.sigmoid(logits))
                }
            elif task == 'regression':
                predictions = {"vlaue": logits}
            else:
                raise ValueError('task 只能设置为cls或者reg')

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)
        return output_spec

    return model_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)
