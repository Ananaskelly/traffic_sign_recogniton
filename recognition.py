import tensorflow as tf
import cv2
import os
import csv
import numpy as np
import img_processing as ip

sess = tf.Session()
saver = tf.train.import_meta_graph('./model2/trained_model.meta')
saver.restore(sess, './model2/trained_model')


def get_class_no(img):
    inp = ip.load_and_process_image(img, False)
    inp = inp.reshape(1, 32, 32, 3)
    prediction = tf.get_default_graph().get_tensor_by_name("softmax:0")
    a = sess.run(prediction, feed_dict={"x:0": inp})
    return np.argmax(a)


"""path = 'D:/rtsd_r3/test'
main = 'D:/rtsd_r3'
CSV_NAME_TEST = 'gt_test.csv'
with open(os.path.join(main, CSV_NAME_TEST), 'r', newline='') as csv_file:
    data_reader = csv.reader(csv_file, delimiter=',')
    it = iter(data_reader)
    next(it)
    for row in it:
        filename = row[0]
        class_no = row[1]
        image = cv2.imread(os.path.join(path, filename), 1)
        rec_class_no = get_class_no(image)
        # cv2.imshow(str(class_no) + ' ' + str(rec_class_no), image)
        print(str(class_no) + ' ' + str(rec_class_no))
"""