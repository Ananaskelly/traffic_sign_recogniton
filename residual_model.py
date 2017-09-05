import tensorflow as tf


def get_residual(x):
    conv_1 = tf.layers.conv2d(x, filters=64, kernel_size=1, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv_2 = tf.layers.conv2d(conv_1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv_3 = tf.layers.conv2d(conv_2, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())

    return conv_3


def get_model(x, num_classes, num_channels=3):
    if num_channels == 1:
        x = tf.reshape(x, [-1, 32, 32, 1])

    conv_init = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=1, activation=tf.nn.relu, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())

    layer_1 = tf.add(conv_init, get_residual(conv_init))
    layer_2 = tf.add(layer_1, get_residual(layer_1))
    layer_3 = tf.add(layer_2, get_residual(layer_2))
    layer_4 = tf.add(layer_3, get_residual(layer_3))
    layer_5 = tf.add(layer_4, get_residual(layer_4))

    fc_input = tf.reshape(layer_5, [-1, 8*8*128])
    fc_1 = tf.layers.dense(fc_input, units=1024, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1, units=num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           activation=None)

    return fc_2
