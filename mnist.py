import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X1 = tf.placeholder(tf.float32, [None, 784])
X2 = tf.placeholder(tf.float32, [None, 49])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable('W1', shape = [784, 49], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape = [49, nb_classes], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([49]))
b2 = tf.Variable(tf.random_normal([nb_classes]))

X2 = tf.nn.relu(tf.matmul(X1, W1) + b1)
hypothesis = tf.nn.softmax(tf.sigmoid(tf.matmul(X2, W2) + b2))

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100
saver = tf.train.Saver()

min_loss = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X1: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch : %04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
        acc, loss = sess.run([accuracy, cost], feed_dict = {X1: mnist.validation.images, Y: mnist.validation.labels})
        print("validation acc : {:.2%}, loss : {:.5f}".format(acc, loss))
        if min_loss > loss:
            min_loss = loss
            saver.save(sess, "./save.ckpt")
            print("Session saved")
        print("")

    print("Learning finished")

    saver.restore(sess, "./save.ckpt")
    print("Train Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.train.images, Y: mnist.train.labels})))
    print("Valid Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.validation.images, Y: mnist.validation.labels})))
    print("Test Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.test.images, Y: mnist.test.labels})))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X1: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
