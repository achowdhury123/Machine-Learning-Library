import tensorflow as tf
import numpy as np

class GRUNeuralNetwork:

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

    class GRUCell:

        def __init__(self, n_inputs, n_neurons):

            sigma = np.sqrt(2) * np.sqrt(2/(n_inputs + n_neurons))

            self.W_xz = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=sigma))
            self.W_xr = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=sigma))
            self.W_xg = tf.Variable(tf.truncated_normal((n_inputs, n_neurons), stddev=sigma))
            self.W_hz = tf.Variable(tf.truncated_normal((n_neurons, n_neurons), stddev=sigma))
            self.W_hr = tf.Variable(tf.truncated_normal((n_neurons, n_neurons), stddev=sigma))
            self.W_hg = tf.Variable(tf.truncated_normal((n_neurons, n_neurons), stddev=sigma))

        def timeStep(self, X, h_last=None):

            if h_last == None:

                z = tf.sigmoid(tf.matmul(X, self.W_xz))
                r = tf.sigmoid(tf.matmul(X, self.W_xr))
                g = tf.tanh(tf.matmul(X, self.W_xg) + tf.matmul(r, self.W_hg))
                h = (1-z) + z * g
                y = (1-z) + z * g

                return h, y

            else:

                z = tf.sigmoid(tf.matmul(X, self.W_xz) + tf.matmul(h_last, self.W_hz))
                r = tf.sigmoid(tf.matmul(X, self.W_xr) + tf.matmul(h_last, self.W_hr))
                g = tf.tanh(tf.matmul(X, self.W_xg) + tf.matmul(r * h_last, self.W_hg))
                h = (1-z) * h_last + z * g
                y = (1-z) * h_last + z * g

                return h, y

        def calculate_sequence(self, X, rows):

            outputs = []
            short_term_states = []

            for i in range(X.get_shape()[0]):
                if i == 0:
                    h, output = self.timeStep(tf.reshape(tf.slice(X, [i, 0, 0], [1, tf.shape(X)[1], tf.shape(X)[2]]), [tf.shape(X)[1], tf.shape(X)[2]]))
                    outputs.append(output)
                    short_term_states.append(h)
                else:
                    h, output = self.timeStep(tf.reshape(tf.slice(X, [i, 0, 0], [1, tf.shape(X)[1], tf.shape(X)[2]]), [tf.shape(X)[1], tf.shape(X)[2]]), h_last=short_term_states[i-1])
                    outputs.append(output)
                    short_term_states.append(h)

            return tf.convert_to_tensor(outputs), tf.convert_to_tensor(short_term_states)

    def calculate_layers(self, X, layers):

        gru_cells = []
        outputs = []
        short_term_states = []

        for i in range(len(layers)-1):

            gru_cells.append(self.GRUCell(layers[i], layers[i+1]))

        output, short_term_state = gru_cells[0].calculate_sequence(X, layers[0])

        outputs.append(output)
        short_term_states.append(short_term_state)

        if len(layers)-1 > 1:

            for i in range(len(layers)-2):

                output, short_term_state = gru_cells[i+1].calculate_sequence(outputs[i], layers[i + 1])

                outputs.append(output)
                short_term_states.append(short_term_state)

        return tf.convert_to_tensor(output), short_term_states

    def __init__(self, hidden_layer_list):
        self.hidden_layer_list = hidden_layer_list

    def train(self, X_train, y_train, learning_rate, iterations, batch_size, optimizer=tf.train.AdamOptimizer, structure="seq_to_vec", sequence_length=None, output_layer=None):

        self.structure = structure
        self.sequence_length = sequence_length

        if self.structure == "seq_to_seq":
            X = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])
            y = tf.placeholder(tf.float32, [self.sequence_length, 1, 1])

            final_outputs, final_short_term_states = self.calculate_layers(X, self.hidden_layer_list)

            loss = tf.reduce_mean(tf.square(final_outputs - y))
            optimizer = optimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)

        else:
            X = tf.placeholder(tf.float32, [X_train.shape[1]/self.hidden_layer_list[0], None, self.hidden_layer_list[0]])
            y = tf.placeholder(tf.int32, [None])

            final_outputs, final_short_term_states = self.calculate_layers(X, self.hidden_layer_list)

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

            final_outputs, final_short_term_states = self.calculate_layers(X, self.hidden_layer_list)

        else:

            X = tf.placeholder(tf.float32, [X_test.shape[1]/self.hidden_layer_list[0], None, self.hidden_layer_list[0]])

            final_outputs, final_short_term_states = self.calculate_layers(X, self.hidden_layer_list)

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
