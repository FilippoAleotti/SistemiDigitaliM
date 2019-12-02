import tensorflow as tf

class Network(object):
    def __init__(self, x, labels=None, is_training=True):
        self.is_training = is_training
        self.input = x
        self.labels = labels
        self.loss = None
        self.output = None

        self._build_network()
        
        if not self.is_training:
            return 
        
        self._build_loss()

    def _build_network(self):
        ''' Build the network '''
        with tf.variable_scope('build_network'):
            batch = self.input.get_shape().as_list()[0]
            c1 = self.conv2d(self.input, kernel_shape=[5,5,1,6], bias_shape=[6], name='c1')
            c1 = tf.nn.tanh(c1)
            s2 = tf.nn.max_pool(c1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='s2')
            s2 = tf.nn.tanh(s2)
            c3 = self.conv2d(s2, kernel_shape=[5,5,6,16], bias_shape=[16], name='c3')
            c3 = tf.nn.tanh(c3)
            s4 = tf.nn.max_pool(c3,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='s4')
            s4 = tf.nn.tanh(s4)
            s4 = tf.reshape(s4,[batch, -1])
            c5 = tf.contrib.layers.fully_connected(s4, num_outputs=120 , activation_fn=tf.nn.tanh)
            f6 = tf.contrib.layers.fully_connected(c5, num_outputs=84 ,  activation_fn=tf.nn.tanh)
            self.output = tf.contrib.layers.fully_connected(f6, num_outputs=10, activation_fn=None)
            self.prediction = tf.squeeze(tf.argmax(self.output, axis=-1))
            
    def _build_loss(self):
        ''' Get the loss value '''
        with tf.variable_scope('build_loss'):
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy_loss)

    def conv2d(self, x, kernel_shape, bias_shape, strides=1, padding='VALID', name='conv2D'):
        ''' Block for 2D Convolution '''
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            biases = tf.get_variable("biases", bias_shape, initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
            output = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padding)
            output = tf.nn.bias_add(output, biases)
            return output