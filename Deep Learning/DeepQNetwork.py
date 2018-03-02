import numpy as np
import tensorflow as tf

class DeepQNetwork:

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

    def neuron_filter(self, height, width):
        stddev = 2/np.sqrt(height * width)
        init = tf.truncated_normal((height, width), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        return W

    def zero_padding(self, X, strides):

        width = int(X.get_shape()[2]) % strides[1]
        height = int(X.get_shape()[1]) % strides[0]

        if width == 0:
            padding = tf.constant([[0, 0], [0, 0], [0, 1]])
            X = tf.pad(X, padding, "CONSTANT")
        else:
            for i in range(width):
                if width % 2 == 0:
                    padding = tf.constant([[0, 0], [0, 0], [0, 1]])
                    X = tf.pad(X, padding, "CONSTANT")
                else:
                    padding = tf.constant([[0, 0], [0, 0], [1, 0]])
                    X = tf.pad(X, padding, "CONSTANT")

        if height == 0:
            padding = tf.constant([[0, 0], [0, 1], [0, 0]])
            X = tf.pad(X, padding, "CONSTANT")
        else:
            for i in range(height):
                if width % 2 == 0:
                    padding = tf.constant([[0, 0], [0, 1], [0, 0]])
                    X = tf.pad(X, padding, "CONSTANT")
                else:
                    padding = tf.constant([[0, 0], [1, 0], [0, 0]])
                    X = tf.pad(X, padding, "CONSTANT")

        return X

    def conv_layer(self, X, filter_num, kernel, strides, activation=None):

        X = self.zero_padding(X, strides)

        filters = []
        for i in range(filter_num):
            filters.append(self.neuron_filter(kernel[0], kernel[1]))

        output_layers = []
        for filter_layer in filters:
            outputs = []
            for i in range(int((int(X.get_shape()[1]) - kernel[0]) / strides[0] + 1)):
                for j in range(int((int(X.get_shape()[2]) - kernel[1]) / strides[1] + 1)):
                    X_temp = 0
                    for k in range(X.get_shape()[0]):
                        X_new = tf.slice(X, [k, i * strides[0], j * strides[1]], [1, kernel[0], kernel[1]])
                        filter_product = filter_layer * X_new
                        X_temp = X_temp + filter_product
                    outputs.append(tf.reduce_sum(X_temp))
            outputs = tf.convert_to_tensor(outputs)
            output_layers.append(tf.reshape(outputs, [int((int(X.get_shape()[1]) - kernel[0]) / strides[0] + 1), int((int(X.get_shape()[2]) - kernel[1]) / strides[1] + 1)]))
        output_layers = tf.convert_to_tensor(output_layers)

        if activation == None:
            return output_layers
        else:
            return activation(output_layers)

    def pooling_layer(self, X, kernel, strides):

        X = self.zero_padding(X, strides)

        output_layers = []
        for filter_layer in range(X.get_shape()[0]):
            outputs = []
            for i in range(int((int(X.get_shape()[1]) - kernel[0]) / strides[0] + 1)):
                for j in range(int((int(X.get_shape()[2]) - kernel[1]) / strides[1] + 1)):
                    X_temp = tf.slice(X, [filter_layer, i * strides[0], j * strides[1]], [1, kernel[0], kernel[1]])
                    outputs.append(tf.reduce_max(X_temp))
            outputs = tf.convert_to_tensor(outputs)
            output_layers.append(tf.reshape(outputs, [int((int(X.get_shape()[1]) - kernel[0]) / strides[0] + 1), int((int(X.get_shape()[2]) - kernel[1]) / strides[1] + 1)]))

        return(tf.convert_to_tensor(output_layers))

    def __init__(self, hidden_layer_list):
        self.hidden_layer_list = hidden_layer_list

    def calculate_layers(self, X, hidden_layer_list, activation):

        hidden_layers = []

        count = 0

        first_layer = self.hidden_layer_list[0]

        hidden_layers.append(self.conv_layer(X, first_layer[1], first_layer[2], first_layer[3]))

        for i in range(len(self.hidden_layer_list)-1):
            new_layer = self.hidden_layer_list[i+1]
            if new_layer[0] == "conv":
                hidden_layers.append(self.conv_layer(hidden_layers[i], new_layer[1], new_layer[2], new_layer[3], activation))
            elif new_layer[0] == "pool":
                hidden_layers.append(self.pooling_layer(hidden_layers[i], new_layer[1], new_layer[2]))
            else:
                if count == 0:
                    hidden_layers.append(self.neuron_layer(tf.reshape(hidden_layers[i], [-1, tf.size(hidden_layers[i])]), new_layer[1], activation))
                    count = count + 1
                else:
                    hidden_layers.append(self.neuron_layer(hidden_layers[i], new_layer[1], activation))

        return hidden_layers

    def exploration_policy(self, Q_values, training_steps, step, action_space):

        epsilon = max(0.05, 1 - step / training_steps)
        probability = np.random.rand()

        if epsilon > probability:
            return np.random.randint(action_space)
        else:
            return np.argmax(Q_values)
