"""CNN for text classification."""
import os
import datetime
import tensorflow as tf
from dataPreprocessing_wip import batch_iterator


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def maxpool2D(x, seq_len, f_size):
    return tf.nn.max_pool(x, ksize=[1, seq_len - f_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')


def length_variable(x):
    return tf.Variable(len(x))


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


class cnn(object):
    """Class for convolutional neural net."""

    def __init__(self, seq_len, n_dim=300, n_classes=2, filter_sizes=[3, 4, 5], n_filters=128):
        """Instanciate the CNN."""
        self.n_classes = n_classes
        with tf.name_scope("Embedding"):
            # x = placeholder for w2v sentences (embedding already done by Google corpus)
            self.x = tf.placeholder(tf.float32, [None, seq_len, n_dim], name="xinput")
            variable_summaries(self.x, 'x')
            # x = tf.placeholder(tf.float32, shape=None)
            self.y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y_input")
            # variable_summaries(self.y_, 'y_')
            self.x_4D = tf.expand_dims(self.x, -1)  # 4D tensor to fit the conv layer input
            # variable_summaries(self.x_4D, 'x_4D')

        with tf.name_scope("Conv2DMaxPool"):
            # Conv2D + Maxpooling layer
            convpool_outputs = []
            for f_size in filter_sizes:
                filter_shape = [f_size, n_dim, 1, n_filters]
                W = weight_variable(filter_shape)
                b = bias_variable([n_filters])
                h_conv = tf.nn.relu(conv2D(self.x_4D, W) + b)
                h_pool = maxpool2D(h_conv, seq_len, f_size)
                convpool_outputs.append(h_pool)
            n_filters_total = n_filters * len(filter_sizes)
            self.h_pool_c = tf.concat(3, convpool_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool_c, [-1, n_filters_total])

        with tf.name_scope("Dropout"):
            # Dropout layer
            self.keep_prob = tf.placeholder(tf.float32)
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        with tf.name_scope("Readout"):
            # Readout layer : dense + softmax to convert raw scores into normalized probabilites
            W_fc = weight_variable([n_filters_total, n_classes])
            b_fc = bias_variable([n_classes])
            self.y = tf.nn.softmax(tf.matmul(self.h_drop, W_fc) + b_fc)
            loss_vect = tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
            self.loss = tf.reduce_mean(loss_vect)
            correct_predictions = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def train(self, x_train, y_train, x_test, y_test, n_epoch=10):
        """Train the cnn."""
        self.session = tf.Session()
        with self.session.as_default():
            optimizer = tf.train.AdamOptimizer(1e-3)
            grad_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grad_vars)
            # summaries
            acc_summary = tf.scalar_summary('accuracy', self.accuracy)
            loss_summary = tf.scalar_summary('loss', self.loss)
            summary_op = tf.merge_summary([acc_summary, loss_summary])
            summary_dir = os.path.join('cnn_logs', 'summaries')
            summary_writer = tf.train.SummaryWriter(summary_dir, self.session.graph)
            # Init session
            self.session.run(tf.initialize_all_variables())
            # Create the batch iterator
            batches = batch_iterator(list(zip(x_train, y_train)), 64, n_epoch)
            # Train loop
            i = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # train step
                feed_dict = {self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5}
                _, summaries, loss, accuracy = self.session.run([train_op, summary_op, self.loss, self.accuracy], feed_dict)
                time = datetime.datetime.now().isoformat()
                i += 1
                print("%s : step %s || loss %s , acc %s" % (time, i, loss, accuracy))
                summary_writer.add_summary(summaries, i)
                # Evaluation on test set every 100 steps
                if i % 100 == 0:
                    print("\nEvaluation on test-set")
                    feed_dict = {self.x: x_test, self.y_: y_test, self.keep_prob: 1.0}
                    _, loss, accuracy = self.session.run([train_op, self.loss, self.accuracy], feed_dict)
                    print("%s : step %s || loss %s , acc %s" % (time, i, loss, accuracy))
                    print("")

    def save(self, filename):
        """Save the session."""
        saver = tf.train.Saver()
        save_path = saver.save(self.session, filename)
        print("\n Model saved in file: %s\n" % save_path)

    def load(self, filename):
        """Restore session."""
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver = saver.restore(self.session, filename)
        print("\n Model restored \n")

    def classify(self, x_input, y_buffer):
        """classify the sentences in x_input."""
        # y_buffer = [[0.0, 0.0] for x in x_input]
        feed_dict = {self.x: x_input, self.y_: y_buffer, self.keep_prob: 1.0}
        predictions = self.session.run(self.y, feed_dict)
        return predictions

    def classify_binary(self, x_input, y_buffer):
        """Convert softmax probability distribution to binary prediction vector."""
        feed_dict = {self.x: x_input, self.y_: y_buffer, self.keep_prob: 1.0}
        predictions = self.session.run(self.y, feed_dict)
        binary_predictions = []
        # [0, 1] = pos && [1, 0] = neg
        for pred in predictions:
            if pred[0] > pred[1]:
                binary_predictions.append('neg')
            elif pred[0] < pred[1]:
                binary_predictions.append('pos')
        return binary_predictions

    def getMaxSentiments(self, x_input, y_buffer):
        """Return the highest probs."""
        feed_dict = {self.x: x_input, self.y_: y_buffer, self.keep_prob: 1.0}
        predictions = self.session.run(self.y, feed_dict)
        flat_y = [(max(p), p.argmax()) for p in predictions]
        max_index_merged = []
        for n in range(self.n_classes):
            max_buffer = 0
            for y in flat_y:
                if y[1] == n:
                    if y[0] > max_buffer:
                        max_buffer = y[0]
                        index = flat_y.index(y)
            max_index_merged.append((y[0], index))
        return max_index_merged
