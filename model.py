import tensorflow as tf

def normal_nn_model(inputs, params):
    images = inputs['images']
    images_shape = images.get_shape().as_list()

    if images_shape != [None, params.image_size, params.image_size, 3]:
        print("input images size error(%s vs %s)" %
              (str(images_shape), str([None, params.image_size, params.image_size, 3])))
        return None

    images = tf.reshape(images, [-1, params.image_size * params.image_size * 3])

    with tf.variable_scope('fc_1'):
        net_l1 = tf.layers.dense(images, 25, activation=tf.nn.relu)
    with tf.variable_scope('fc_2'):
        net_l2 = tf.layers.dense(net_l1, 12, activation=tf.nn.relu)
    with tf.variable_scope('fc_3'):
        logits = tf.layers.dense(net_l2, 2, activation=tf.nn.softmax)

    return logits

def model(mode, inputs, params, reuse=False):
    model_spec = inputs
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    with tf.variable_scope('model', reuse=reuse):
        #if params.model == 'normal':
        logits = normal_nn_model(inputs, params)
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(tf.argmax(labels, 1), predictions)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
        model_spec['train_op'] = train_op

    with tf.variable_scope('metrics'):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op

    return model_spec