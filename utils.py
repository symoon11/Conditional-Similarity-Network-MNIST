import tensorflow as tf


def create_weight_variables(shape, name):

    if len(shape) == 4:
        in_out = shape[0]*shape[1]*shape[2] + shape[3]
    else:
        in_out = shape[0] + shape[1]

    import math
    stddev = math.sqrt(2 / in_out) # XAVIER INITIALIZER

    initializer = tf.truncated_normal(shape, stddev=stddev)

    return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


def create_bias_variables(shape, name):

    initializer = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


def create_mask_variables(shape, name):

    initializer = tf.truncated_normal(shape, mean=0.9, stddev=0.7)
    return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


def create_conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def create_max_pool(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_relu(x, bias):

    return tf.nn.relu(tf.nn.bias_add(x, bias))