# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tensor_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

tensor_list2 = [[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]], [[13, 14, 15, 16]], [[17, 18, 19, 20]]]

with tf.Session() as sess:
    for i in np.arange(1):
        print ('##################### %d'%i)
        x1 = tf.train.batch(tensor_list, batch_size=2, enqueue_many=False, capacity=1)

        x2 = tf.train.batch(tensor_list, batch_size=2, enqueue_many=True, capacity=1)

        y1 = tf.train.batch_join(tensor_list, batch_size=3, enqueue_many=False, capacity=1)

        y2 = tf.train.batch_join(tensor_list2, batch_size=25, enqueue_many=True, capacity=1, allow_smaller_final_batch=False)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("x1 batch:"+"-"*10)
        x1_batch = sess.run(x1)
        print(x1_batch)

        print("x2 batch:"+"-"*10)

        print(sess.run(x2))

        print("y1 batch:"+"-"*10)

        print(sess.run(y1))

        print("y2 batch:"+"-"*10)

        print(sess.run(y2))

        print("-"*10)

    coord.request_stop()

    coord.join(threads)
