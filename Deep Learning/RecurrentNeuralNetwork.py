import numpy as np
import tensorflow as tf

class RecurrentNeuralNetwork:

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

    class RecurrentCell:

        def __init__(self, n_inputs, n_neurons):

            sigma = np.sqrt(2) * np.sqrt(2/(n_inputs + n_neurons))


            self.W_x = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=sigma))
            self.W_y = tf.Variable(tf.truncated_normal((n_neurons, n_neurons), stddev=sigma))
            self.b = tf.zeros([1, n_neurons], dtype=tf.float32)

        def timeStep(self, X, state=None, activation=None):

            if state == None:
                if activation == None:
                    return (tf.matmul(X, self.W_x) + self.b), (tf.matmul(X, self.W_x) + self.b)
                else:
                    return activation(tf.matmul(X, self.W_x) + self.b), activation(tf.matmul(X, self.W_x) + self.b)

            else:
                if activation == None:
                    return (tf.matmul(X, self.W_x) + tf.matmul(state, self.W_y) + self.b), (tf.matmul(X, self.W_x) + tf.matmul(state, self.W_y) + self.b)
                else:
                    return activation(tf.matmul(X, self.W_x) + tf.matmul(state, self.W_y) + self.b), activation(tf.matmul(X, self.W_x) + tf.matmul(state, self.W_y) + self.b)

        def calculate_sequence(self, X):

            outputs = []
            states = []

            for i in range(X.get_shape()[0]):
                if i == 0:
                    output, state = self.timeStep(tf.reshape(tf.slice(X, [i, 0, 0], [1, tf.shape(X)[1], tf.shape(X)[2]]), [tf.shape(X)[1], tf.shape(X)[2]]), activation=tf.nn.relu)
                    outputs.append(output)
                    states.append(state)
                else:
                    output, state = self.timeStep(tf.reshape(tf.slice(X, [i, 0, 0], [1, tf.shape(X)[1], tf.shape(X)[2]]), [tf.shape(X)[1], tf.shape(X)[2]]), state=states[i-1], activation=tf.nn.relu)
                    outputs.append(output)
                    states.append(state)

            return tf.convert_to_tensor(outputs), tf.convert_to_tensor(states)

    def calculate_layers(self, X, layers):

        recurrent_cells = []
        outputs = []
        states = []

        for i in range(len(layers)-1):

            recurrent_cells.append(self.RecurrentCell(layers[i], layers[i+1]))

        output, state = recurrent_cells[0].calculate_sequence(X)

        outputs.append(output)
        states.append(state)

        if len(layers)-1 > 1:

            for i in range(len(layers)-2):

                output, state = recurrent_cells[i+1].calculate_sequence(outputs[i])

                outputs.append(output)
                states.append(state)

        return tf.convert_to_tensor(output), states

    def __init__(self, hidden_layer_list):
        self.hidden_layer_list = hidden_layer_list

    def train(self, X_train, y_train, learning_rate, iterations, batch_size, optimizer=tf.train.AdamOptimizer, structure="seq_to_vec", sequence_length=None, output_layer=None):

        self.structure = structure
        self.sequence_length = sequence_length
        if self.structure == "seq_to_seq":
            X = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])
            y = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])

            final_outputs, final_states = self.calculate_layers(X, self.hidden_layer_list)

            loss = tf.reduce_mean(tf.square(final_outputs - y))
            optimizer = optimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)

        else:
            X = tf.placeholder(tf.float32, [X_train.shape[1]/self.hidden_layer_list[0], None, self.hidden_layer_list[0]])
            y = tf.placeholder(tf.int32, [None])

            final_outputs, final_states = self.calculate_layers(X, self.hidden_layer_list)

            final_outputs = tf.transpose(final_outputs, [1, 0, 2])
            final_outputs = tf.reshape(final_outputs, [-1, X_train.shape[1]])

            logits = self.neuron_layer(final_outputs, output_layer)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

            loss = tf.reduce_mean(xentropy)
            optimizer = optimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()
            for iterations in range(iterations):
                for iteration in range(X_train.shape[0] // batch_size):
                    if structure == "seq_to_seq":
                        random_indices = np.random.randint(X_train.shape[2]-self.sequence_length)
                        X_subset = np.array(y_train[0][0][random_indices:random_indices+20], ndmin=3)
                        y_subset = np.array(y_train[0][0][random_indices+1:random_indices+21], ndmin=3)

                        X_subset = np.reshape(X_subset, (sequence_length, 1, 1))
                        y_subset = np.reshape(y_subset, (sequence_length, 1, 1))

                    else:
                        random_indices = np.random.permutation(X_train.shape[0])[:batch_size]
                        X_subset = X_train[random_indices]
                        y_subset = y_train[random_indices]

                        X_subset = np.reshape(X_subset, (-1, int(X_train.shape[1]/self.hidden_layer_list[0]), int(self.hidden_layer_list[0])))
                        X_subset = np.transpose(X_subset, (1, 0, 2))

                    sess.run(training_op, feed_dict={X: X_subset, y: y_subset})

            save_path = saver.save(sess, "./my_model.ckpt")

        return self

    def predict(self, X_test, output_layer=None):

        tf.reset_default_graph()

        if self.structure == "seq_to_seq":

            X = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])
            y = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])

            final_outputs, final_states = self.calculate_layers(X, self.hidden_layer_list)

        else:

            X = tf.placeholder(tf.float32, [X_test.shape[1]/self.hidden_layer_list[0], None, self.hidden_layer_list[0]])

            final_outputs, final_states = self.calculate_layers(X, self.hidden_layer_list)

            final_outputs = tf.transpose(final_outputs, [1, 0, 2])
            final_outputs = tf.reshape(final_outputs, [-1, X_test.shape[1]])

            logits = self.neuron_layer(final_outputs, output_layer)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, "./my_model.ckpt")

            if self.structure == "seq_to_seq":

                X_test = np.array(X_test, ndmin=3)

                X_test = np.reshape(X_test, (self.sequence_length, 1, 1))

                return final_outputs.eval(feed_dict={X: X_test})

            else:

                X_test = np.reshape(X_test, (-1, int(X_test.shape[1]/self.hidden_layer_list[0]), int(self.hidden_layer_list[0])))
                X_test = np.transpose(X_test, (1, 0, 2))

                y = logits.eval(feed_dict={X: X_test})

                return np.argmax(y)
