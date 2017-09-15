import tensorflow as tf
import os

import russian_dataset
import cnn_base as cnn

data = russian_dataset.read(106)

print("Images loaded..")

learning_rate = 0.001
training_iterations = 300000
batch_size = 28
display_step = 10
beta = 0.001

n_classes = 106
dropout = 0.5

# tf Graph input
x = tf.placeholder(tf.float32, [None, 32, 32, 3], "x")
y = tf.placeholder(tf.float32, [None, n_classes], "y")
keep_prob = tf.placeholder(tf.float32)

prediction = cnn.get_model(x, n_classes)
prediction = tf.identity(prediction, name="prediction")

softmax = tf.nn.softmax(prediction, -1, "softmax")
answer = tf.argmax(softmax, 0, name="answer42")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iterations:
        batch_x, batch_y = data.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.7f}".format(loss) + ", Training Accuracy= " +
                  "{:.7f}".format(acc))
        step += 1

    saver = tf.train.Saver()

    saver_def = saver.as_saver_def()

    saver.save(sess, os.path.join(os.getcwd(), 'trained_model'))

    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={x: data.test.images,
                                      y: data.test.labels,
                                      keep_prob: 1.}))
