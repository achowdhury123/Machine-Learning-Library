import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt

class DenoisingAutoEncoder:

    def batch_normalization(self, X):
        scale = tf.Variable(1.0)
        shift = tf.Variable(0.1)
        mean, variance = tf.nn.moments(X, axes=1)
        normalized_vector = tf.transpose(tf.divide(tf.subtract(tf.transpose(X), mean), tf.sqrt(variance)))
        return normalized_vector * scale + shift


    def neuron_layer(self, X, n_neurons, activation=None, batch_norm=False):
        n_inputs = int(X.get_shape()[1])
        if activation == tf.sigmoid:
            sigma = np.sqrt(2/(n_inputs + n_neurons))
        elif activation == tf.tanh:
            sigma = 4 * np.sqrt(2/(n_inputs + n_neurons))
        else:
            sigma = np.sqrt(2) * np.sqrt(2/(n_inputs + n_neurons))
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=sigma)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if batch_norm == True:
            normalized_values = self.batch_normalization(Z)
        else:
            normalized_values = Z
        if activation is not None:
            return activation(normalized_values)
        else:
            return normalized_values

    def calculate_layers(self, X, hidden_layer_list):

        hidden_layers = []

        X = X + tf.random_normal(tf.shape(X))

        hidden_layers.append(self.neuron_layer(X, hidden_layer_list[0], activation = tf.nn.relu))

        for i in range(len(hidden_layer_list)-1):
            if i == len(hidden_layer_list)-2:
                hidden_layers.append(self.neuron_layer(hidden_layers[i], hidden_layer_list[i+1]))
            else:
                hidden_layers.append(self.neuron_layer(hidden_layers[i], hidden_layer_list[i+1], tf.nn.relu))

        return hidden_layers

    def __init__(self, hidden_layer_list):
        self.hidden_layer_list = hidden_layer_list

    def train(self, X_train, learning_rate, iterations, batch_size, optimizer=tf.train.GradientDescentOptimizer):

        X = tf.placeholder(tf.float32, shape=(None, self.hidden_layer_list[0]), name="X")

        del self.hidden_layer_list[0]

        outputs = self.calculate_layers(X, self.hidden_layer_list)[len(self.hidden_layer_list)-1]

        loss = tf.reduce_mean(tf.square(outputs-X))

        optimizer = optimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()
            for iterations in range(iterations):
                for iteration in range(X_train.shape[0] // batch_size):

                    random_indices = np.random.permutation(X_train.shape[0])[:batch_size]
                    X_subset = X_train[random_indices]

                    sess.run(training_op, feed_dict={X: X_subset})

            save_path = saver.save(sess, "./my_model.ckpt")

        return self

    def predict(self, X_test):

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, shape=(None, X_test.shape[1]), name="X")

        outputs = self.calculate_layers(X, self.hidden_layer_list)[len(self.hidden_layer_list)-1]

        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, "./my_model.ckpt")

            y = outputs.eval(feed_dict={X: X_test})

            return y
