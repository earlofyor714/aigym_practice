import numpy as np
import tensorflow as tf


# Input size: 210 x 160 x 3
# Output size: 6

class LstmAgent(object):
    def __init__(self, input_size, epsilon=1.0, learning_rate=0.5, num_nodes=64, batch_size=1, num_unrollings=10):
        self.num_nodes=num_nodes
        self.batch_size=batch_size
        self.num_rollings=num_unrollings
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.output_size = 6

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

    """Due to Tensorflow session's nature, contains game logic + AI"""
    def tensorflow_learn(self, state, env, tolerance, n_frames, is_display=True):
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            is_quit = False
            time = 1.0
            alpha = 1.0
            current_state = self.build_state(state)

            while self.epsilon > tolerance:
                print("epsilon: {}".format(self.epsilon))
                for _ in range(n_frames):
                    try:
                        # From state, get action
                        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.lstm_input: current_state})
                        if np.random.rand(1) < self.epsilon:
                            a[0] = env.get_action_space().sample()
                        # From action, get next state and reward
                        next_state, reward, env.done, _ = env.act(a[0])
                        next_state = self.build_state(next_state)
                        # From next state, get most probable next action
                        Q1 = sess.run([self.Qout], feed_dict={self.lstm_input: next_state})
                        maxQ1 = np.max(Q1)
                        targetQ = allQ
                        targetQ[0, a[0]] = reward + alpha * maxQ1
                        _, mem_state = sess.run([self.optimizer, self.saved_state],
                                                feed_dict={self.lstm_input: current_state, self.nextQ: targetQ})
                        current_state = next_state

                        env.trial_data['final_time'] += 1
                        env.trial_data['net_reward'] += reward
                    except KeyboardInterrupt:
                        is_quit = True
                    finally:
                        if time >= (n_frames-1):
                            env.trial_data['success'] = True
                        if is_quit or env.done:
                            break
                    if is_display:
                        env.render()
                self.epsilon = 1.0 / (time * time)
                # time += 0.001
                time += 1.0
                env.reset(False)

        return self.epsilon, is_quit

    def choose_action(self, state):
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            current_state = self.build_state(state)
            a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.lstm_input: current_state})
            return a[0]

    def build_state(self, in_state):
        mean_state = np.mean(in_state, axis=2)
        matrix_state = np.divide(mean_state, 256)
        state = [matrix_state.flatten()]
        return state


# Actions:
#   0 = stop/left, 1 = fire, 2 = right, 3 = stop/left, 4 = right+fire, 5 = left+fire

# TODO: move training loop into here; refactor testing loop within simulator
