

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import tempfile

import tensorflow as tf

from data_providers import DataProvider

FLAGS = None


class ConvolutionNN:
    def __init__(self, batch_size=100, file_name='ocr_conv_nn', path='ocr_model/'):
        """Constructor of the Convolutional Neural Network model

        Args:
            An integer (batch_size) indicating the size of the batches.
            Two string (file_name and path) indicating the file name and the path
            where the model will be saved.
        """
        # Load the data
        self.train_data = DataProvider(batch_size, which_set='train')
        self.test_data = DataProvider(batch_size, which_set='test')
        self.n_classes = self.train_data.num_classes

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])

        # Build the graph for the deep net
        self.y_conv, self.keep_prob = self.build()

        # Path where the file will be saved
        self.savefile = path + file_name

    def build(self):
        """build() builds the graph for a deep net for classifying digits.

        Args:
            None

        Returns:
            A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
            equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9). keep_prob is a scalar placeholder for the probability of
            dropout.
        """

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_variable([5, 5, 1, 32], var_name='W_conv1')
            self.b_conv1 = self.bias_variable([32], var_name='b_conv1')
            h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_variable([5, 5, 32, 64], var_name='W_conv2')
            self.b_conv2 = self.bias_variable([64], var_name='b_conv2')
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], var_name='W_fc1')
            self.b_fc1 = self.bias_variable([1024], var_name='b_fc1')

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_variable([1024, self.n_classes], var_name='W_fc2')
            self.b_fc2 = self.bias_variable([self.n_classes], var_name='b_fc2')

            y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
            #labels = tf.argmax(y_conv, 1)

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, self.n_classes])

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv)
            # TRY: cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        with tf.name_scope('saver'):
            self.saver = tf.train.Saver(
            {'W_conv1': self.W_conv1, 'b_conv1': self.b_conv1,
             'W_conv2': self.W_conv2, 'b_conv2': self.b_conv2,
             'W_fc1': self.W_fc1, 'b_fc1': self.b_fc1,
             'W_fc2': self.W_fc2, 'b_fc2': self.b_fc2
             })

        return y_conv, keep_prob

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape, var_name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=var_name)

    def bias_variable(self, shape, var_name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=var_name)

    def train(self, n_epochs=100):
        """Train the model and save it to self.savefile.

        Args:
            An integer (n_epochs) defining the number of epochs
        """

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(n_epochs):
                i = 0
                start_time = time.time()
                for input_batch, target_batch in self.train_data:
                    train_accuracy = self.accuracy.eval(
                        feed_dict={self.x: input_batch, self.y_: target_batch, self.keep_prob: 1.0})
                    print('epoch {0}, batch {1:03d} - training accuracy: {2:.3f}'.format(e, i, train_accuracy))
                    self.train_step.run(feed_dict={self.x: input_batch, self.y_: target_batch, self.keep_prob: 0.5})
                    i += 1
                print('\t\tepoch {0} completed in {1:.2f} seconds\n'.format(e, time.time() - start_time))

                print('\t\ttest accuracy: %g' % self.accuracy.eval(feed_dict={
                        self.x: self.test_data.inputs, self.y_: self.test_data.targets, self.keep_prob: 1.0}))

            self.saver.save(sess, self.savefile)

    def predict(self, input_sample):
        with tf.Session() as sess:
            # restore the model
            self.saver.restore(sess, self.savefile)
            prediction = sess.run(self.y_conv, feed_dict={self.x: [input_sample], self.keep_prob: 1})

        return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batches', type=int, default=100,
                        dest='batch_size', help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        dest='epochs', help='Number of epochs')
    args, unparsed = parser.parse_known_args()

    CNN = ConvolutionNN(batch_size=args.batch_size)
    CNN.train(n_epochs=args.epochs)
    data = DataProvider(100, which_set='test')
    print(CNN.predict(data.inputs[0]))
