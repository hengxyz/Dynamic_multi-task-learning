import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(decay=0.9)
m = ema.apply([w])
av = ema.average(w)

x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])
y_ = tf.multiply(x, w)

with tf.control_dependencies([m]):
    loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _, w_, av_ = sess.run([train, w, av], feed_dict={x: [1], y: [10]})
        print(w_, ',', av_)