import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from urllib.request import urlretrieve

from Invader.invaders.agents import reader


class LstmAgent:
    def __init__(self, num_nodes=64, batch_size=64, num_unrollings=10):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        self.vocabulary_size = len(string.ascii_lowercase) + 1  # for words

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Parameters:
            # Input gate: input, previous output, and bias
            in_input = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.num_nodes], -0.1, 0.1))
            in_p_out = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
            in_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Forget gate: input, previous output, and bias
            fgt_input = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.num_nodes], -0.1, 0.1))
            fgt_p_out = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
            fgt_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Memory cell: input, state and bias
            mem_input = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.num_nodes], -0.1, 0.1))
            mem_state = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
            mem_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Output gate: input, previous output, and bias
            out_input = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.num_nodes], -0.1, 0.1))
            out_p_out = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
            out_bias = tf.Variable(tf.zeros([1, self.num_nodes]))
            # Variables saving state across unrollings
            self.saved_output = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)
            self.saved_state = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)
            # Classifier weights and biases
            self.w = tf.Variable(tf.truncated_normal([self.num_nodes, self.vocabulary_size], -0.1, 0.1))
            self.b = tf.Variable(tf.zeros([self.vocabulary_size]))

            def lstm_cell(input, output, state):
                """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
                    Note that in this formulation, we omit the various connections between the
                    previous state and the gates."""
                input_gate = tf.sigmoid(tf.matmul(input, in_input) + tf.matmul(output, in_p_out) + in_bias)
                forget_gate = tf.sigmoid(tf.matmul(input, fgt_input) + tf.matmul(output, fgt_p_out) + fgt_bias)
                update = tf.matmul(input, mem_input) + tf.matmul(output, mem_state) + mem_bias
                state = forget_gate * state + input_gate * tf.tanh(update)
                output_gate = tf.sigmoid(tf.matmul(input, out_input) + tf.matmul(output, out_p_out) + out_bias)
                return output_gate * tf.tanh(state), state

            # Input data
            self.train_data = list()
            for _ in range(self.num_unrollings + 1):
                self.train_data.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.vocabulary_size]))
            train_inputs = self.train_data[:num_unrollings]
            train_labels = self.train_data[1:] # Labels are inputs shifted by one time step

            # Unrolled LSTM Loop
            outputs = list()
            output = self.saved_output
            state = self.saved_state
            for i in train_inputs:
                output, state = lstm_cell(i, output, state)
                outputs.append(output)

            # State saving across unrollings
            with tf.control_dependencies([self.saved_output.assign(output),
                                          self.saved_state.assign(state)]):
                #Classifier
                logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), self.w, self.b)
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(train_labels, 0), logits=logits)
                )

            # Optimizer
            global_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

            # Predictions
            self.train_prediction = tf.nn.softmax(logits)

            # Sampling and validation eval: batch 1, no unrolling
            self.sample_input = tf.placeholder(tf.float32, shape=[1, self.vocabulary_size])
            saved_sample_output = tf.Variable(tf.zeros([1, self.num_nodes]))
            saved_sample_state = tf.Variable(tf.zeros([1, self.num_nodes]))
            self.reset_sample_state = tf.group(
                saved_sample_output.assign(tf.zeros([1, self.num_nodes])),
                saved_sample_state.assign(tf.zeros([1, self.num_nodes]))
            )
            sample_output, sample_state = lstm_cell(self.sample_input, saved_sample_output, saved_sample_state)
            with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):
                self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, self.w, self.b))

    def learn(self, train_batches, num_steps=7001, summary_frequency=100):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            mean_loss = 0
            for step in range(num_steps):
                batches = train_batches.next()
                feed_dict = dict()
                for i in range(self.num_unrollings + 1):
                    feed_dict[self.train_data[i]] = batches[i]
                _, l, predictions, lr = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.learning_rate], feed_dict=feed_dict)
                mean_loss += 1
                if step % summary_frequency == 0:
                    if step > 0:
                        mean_loss = mean_loss / summary_frequency
                    # The mean loss is an estimate of the loss over the last few batches
                    print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                    mean_loss = 0
                    labels = np.concatenate(list(batches)[1:])
                    print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
                    if step % (summary_frequency * 10) == 0:
                        # Generate some samples
                        print('=' * 80)
                        for _ in range(5):
                            feed = sample(random_distribution())
                            sentence = characters(feed)[0]
                            self.reset_sample_state.run()
                            for _ in range(79):
                                prediction = self.sample_prediction.eval({self.sample_input: feed})
                                feed = sample(prediction)
                                sentence += characters(feed)[0]
                            print(sentence)
                        print('=' * 80)
                    # Measure validation set perplexity
                        self.reset_sample_state.run()
                        valid_logprob = 0
                        for _ in range(valid_size):
                            b = validation_batches.next()
                            predictions = self.sample_prediction.eval({self.sample_input: b[0]})
                            valid_logprob = valid_logprob + logprob(predictions, b[1])
                        print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


# =========================================
# code custom to word problem
# =========================================
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data


text = read_data(filename)

# train_text, valid_text, test_text, _ = reader.ptb_raw_data('../../../../models/simple-examples/data')
# valid_size = len(valid_text)
# train_size = len(train_text)

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

training_batches = BatchGenerator(train_text, batch_size=64, num_unrollings=10)
validation_batches = BatchGenerator(valid_text, 1, 1)


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:,None]


if __name__ == "__main__":
    agent = LstmAgent()
    agent.learn(training_batches)
