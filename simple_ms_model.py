import tensorflow as tf


def get_model(inp, out_classes):

    inp = tf.reshape(inp, [-1, 32, 32, 3])

    conv_1 = tf.layers.conv2d(inputs=inp, filters=32, kernel_size=5, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=64, kernel_size=5, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

    conv_3 = tf.layers.conv2d(inputs=max_pool_2, filters=128, kernel_size=3, padding='valid',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

    fc_inp_1 = tf.reshape(conv_3, [-1, 3*3*128])
    fc_inp_2 = tf.reshape(conv_2, [-1, 10*10*64])
    fc_inp = tf.concat([fc_inp_1, fc_inp_2], 1)

    fc_1 = tf.layers.dense(inputs=fc_inp, units=1024, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           activation=tf.nn.relu)
    fc_2 = tf.layers.dense(inputs=fc_1, units=out_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())

    return fc_2
