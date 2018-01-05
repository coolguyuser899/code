import tensorflow as tf

# sess = tf.Session()    #session must be closed, or use 'with' statement
# ...
# sess.close()

# a = tf.constant(1.0)
# b = tf.Variable(2.0, name="test_var")
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(a))
#     print(sess.run(b))
#
# graph = tf.get_default_graph()
# for op in graph.get_operations():
#     print(op.name)
#
# a = tf.placeholder("float")
# b = tf.placeholder("float")
# y = tf.multiply(a, b)
#
# feed_dict = {a:2, b:3}
# with tf.Session() as sess:
#     print(sess.run(y, feed_dict))

# b = tf.Variable([10, 20,30,40,50,60], name = 't')   #calculate mean of an array
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.reduce_mean(b)))
#
# a = [ [0.1, 0.2, 0.3],
#       [20, 2, 3],
#       [20, 2, 30]
#     ]
# b = tf.Variable(a, name='b')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.argmax(b, 1)))    #print index of max value in each row


####linear regression, try to fix data on a straight line

import tensorflow as tf
import numpy as np

#create training data
trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

#linear regression model is y = w * x
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)
cost = tf.pow(Y-y_model, 2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))    #about 3.0 

#new session will print value of 0.0
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w))

