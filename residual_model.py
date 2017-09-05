import tensorflow as tf


def get_residual(x):
    conv_1 = tf.layers.conv2d(x, filters=64, kernel_size=1, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv_2 = tf.layers.conv2d(conv_1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv_3 = tf.layers.conv2d(conv_2, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())

    return conv_3


def pad(x, val):

    return tf.pad(x, [[0, 0], [val, val], [val, val], [0, 0]])


def get_model(x, num_classes, num_channels=3):
    if num_channels == 1:
        x = tf.reshape(x, [-1, 32, 32, 1])

    conv_init = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=1, activation=tf.nn.relu, padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

    layer_1 = tf.add(conv_init, pad(get_residual(conv_init), 2))
    layer_2 = tf.add(layer_1, pad(get_residual(layer_1), 2))
    layer_2 = tf.layers.max_pooling2d(layer_2, 2, 2)
    layer_3 = tf.add(layer_2, pad(get_residual(layer_2), 2))
    layer_4 = tf.add(layer_3, pad(get_residual(layer_3), 2))
    layer_4 = tf.layers.max_pooling2d(layer_4, 2, 2)
    layer_5 = tf.add(layer_4, pad(get_residual(layer_4), 2))
    layer_6 = tf.add(layer_5, pad(get_residual(layer_5), 2))

    fc_input = tf.reshape(layer_6, [-1, 7*7*128])
    fc_1 = tf.layers.dense(fc_input, units=1024, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1, units=num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           activation=None)

    return fc_2
