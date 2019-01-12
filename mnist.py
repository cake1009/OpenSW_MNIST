import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 1

train_label = np.argmax(mnist.train.labels, 1)
train_labels = train_label.reshape(-1, 1)
validation_label = np.argmax(mnist.validation.labels, 1)
validation_labels = validation_label.reshape(-1, 1)
test_label = np.argmax(mnist.test.labels, 1)
test_labels = test_label.reshape(-1, 1)

X1 = tf.placeholder(tf.float32, [None, 784])
X2 = tf.placeholder(tf.float32, [None, 392])
X3 = tf.placeholder(tf.float32, [None, 196])
X4 = tf.placeholder(tf.float32, [None, 98])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.get_variable('W1', shape = [784, 392], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape = [392, 196], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape = [196, 98], initializer = tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('W4', shape = [98, nb_classes], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([392]))
b2 = tf.Variable(tf.random_normal([196]))
b3 = tf.Variable(tf.random_normal([98]))
b4 = tf.Variable(tf.random_normal([nb_classes]))

X2 = tf.nn.relu(tf.matmul(X1, W1) + b1)
X3 = tf.nn.relu(tf.matmul(X2, W2) + b2)
X4 = tf.nn.relu(tf.matmul(X3, W3) + b3)
hypothesis = tf.nn.relu(tf.matmul(X4, W4) + b4)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)

is_correct = tf.cast(tf.abs(hypothesis - Y) <= 0.5, dtype = tf.float32)
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
            batch_xs = mnist.train.images[i*batch_size:(i+1)*batch_size, :]
            batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]
            c, _ = sess.run([cost, optimizer], feed_dict={X1: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch : %04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        
        acc, loss = sess.run([accuracy, cost], feed_dict = {X1: mnist.validation.images, Y: validation_labels})
        print("validation acc : {:.2%}, loss : {:.5f}".format(acc, loss))
        if min_loss > loss:
            min_loss = loss
            saver.save(sess, "./save.ckpt")
            print("Session saved")
        print("")

    print("Learning finished")

    saver.restore(sess, "./save.ckpt")
    print("Train Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.train.images, Y: train_labels})))
    print("Valid Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.validation.images, Y: validation_labels})))
    print("Test Accuracy: {:.2%}".format(accuracy.eval(session=sess, feed_dict={X1: mnist.test.images, Y: test_labels})))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X1: mnist.test.images[r:r + 1]}))

    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
