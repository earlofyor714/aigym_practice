import numpy as np
import tensorflow as tf


# Input size: 210 x 160 x 3
# Output size: 6

class LstmAgent(object):
    def __init__(self, input_size, graph=None, alpha=1.0, learning_rate=0.5,
                 num_nodes=64, batch_size=1):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.output_size = 6

        if graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()

        with self.graph.as_default():
            # Inner workings of LSTM
            self.lstm_input = tf.placeholder(shape=[self.batch_size, self.input_size], dtype=tf.float32)

            self.saved_output = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)
            self.saved_state = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)

            # Input gate: input, previous output, and bias
            self.in_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Forget gate: input, previous output, and bias
            self.fgt_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Memory cell: input, state and bias
            self.mem_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Output gate: input, previous output, and bias
            self.out_bias = tf.Variable(tf.zeros([1, self.num_nodes]))

            self.input_weights = tf.Variable(tf.truncated_normal([self.input_size, 4 * self.num_nodes], -0.1, 0.1))
            self.output_weights = tf.Variable(tf.truncated_normal([self.num_nodes, 4 * self.num_nodes], -0.1, 0.1))

            def lstm_cell(lstm_input, lstm_output, lstm_state):
                ifmo_input = tf.matmul(lstm_input, self.input_weights)
                ifmo_output = tf.matmul(lstm_output, self.output_weights)

                fidx = self.num_nodes
                uidx = 2 * self.num_nodes
                oidx = 3 * self.num_nodes
                input_gate = tf.sigmoid(ifmo_input[:, :fidx] + ifmo_output[:, :fidx] + self.in_bias)
                forget_gate = tf.sigmoid(ifmo_input[:, fidx:uidx] + ifmo_output[:, fidx:uidx] + self.fgt_bias)
                update = ifmo_input[:, uidx:oidx] + ifmo_output[:, uidx:oidx] + self.mem_bias
                output_gate = tf.sigmoid(ifmo_input[:, oidx:] + ifmo_output[:, oidx:] + self.out_bias)

                lstm_state = forget_gate * lstm_state + input_gate * tf.tanh(update)
                return output_gate * tf.tanh(lstm_state), lstm_state

            # Variables used for learning LSTM
            self.w = tf.Variable(tf.truncated_normal([self.num_nodes, self.output_size], -0.1, 0.1))
            self.b = tf.Variable(tf.zeros([self.output_size]))

            output = self.saved_output
            state = self.saved_state
            output, state = lstm_cell(self.lstm_input, output, state)
            self.nextQ = tf.placeholder(shape=[1, self.output_size], dtype=tf.float32)
            with tf.control_dependencies([self.saved_output.assign(output),
                                          self.saved_state.assign(state)]):
                self.Qout = tf.nn.xw_plus_b(tf.concat(self.saved_output, 0), self.w, self.b)
                self.predict = tf.argmax(self.Qout, 1)
                loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

    def choose_action(self, current_state, session):
        a, allQ = session.run([self.predict, self.Qout], feed_dict={self.lstm_input: current_state})
        return a[0], allQ

    def learn(self, state, action, allQ, reward, next_state, sess=None):
        if not sess:
            return
        Q1 = sess.run([self.Qout], feed_dict={self.lstm_input: next_state})
        maxQ1 = np.max(Q1)
        targetQ = allQ
        targetQ[0, action] = reward + self.alpha * maxQ1
        _, mem_state = sess.run([self.optimizer, self.saved_state],
                                feed_dict={self.lstm_input: state, self.nextQ: targetQ})

# Actions:
#   0 = stop/left, 1 = fire, 2 = right, 3 = stop/left, 4 = right+fire, 5 = left+fire

# TODO: save weights after training to save time

# TODO: randomize the action returned if allQ has multiple values with same value (i.e. fix self.predict)
