import tensorflow as tf
import numpy as np
import time

#%%
# Python optimisation variables
learning_rate = 0.5
epochs = 100
batch_size = 100
print("batch_size:", batch_size," epochs:", epochs, " learning_rate:", learning_rate)
#%%
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

hiddenLen = 300
# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, hiddenLen], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([hiddenLen]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([hiddenLen, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

W1l = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(W1*W1)))/784/hiddenLen
W2l = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(W2*W2)))/10/hiddenLen
b1l = tf.reduce_sum(b1*b1)/hiddenLen
b2l = tf.reduce_sum(b2*b2)/10

loss = cross_entropy + 0.1*(W1l + W2l + b1l + b2l)

# add an optimiser
#optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   
   t = time.time()
   
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
   print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
   
   elapsed = time.time() - t
      
   print("time",elapsed)
   