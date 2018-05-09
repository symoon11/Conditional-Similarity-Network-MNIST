# some borrowed from Tiago Freitas Pereira's code
from csnet_utils import *


class CSNet(object):

    def __init__(self,
                 conv1_kernel_size = 5,
                 conv1_output = 16,

                 conv2_kernel_size = 5,
                 conv2_output = 32,

                 fc1_output = 400,
                 dim_embedding = 10,

                 num_features = 2):

        self.W_conv1 = create_weight_variables([conv1_kernel_size, conv1_kernel_size, 3, conv1_output], name="W_conv1")
        self.b_conv1 = create_bias_variables([conv1_output], name="b_conv1")

        self.W_conv2 = create_weight_variables([conv1_kernel_size, conv2_kernel_size, conv1_output, conv2_output], name="W_conv2")
        self.b_conv2 = create_bias_variables([conv2_output], name="b_conv2")

        self.W_fc1 = create_weight_variables([(28//4)*(28//4)*conv2_output, fc1_output], name="W_fc1")
        self.b_fc1 = create_bias_variables([fc1_output], name="bias_fc1")

        self.W_fc2 = create_weight_variables([fc1_output, dim_embedding], name="W_fc2")
        self.b_fc2 = create_bias_variables([dim_embedding], name="bias_fc2")

        self.M = create_mask_variables([dim_embedding, num_features], name="mask")

        self.dim_embedding = dim_embedding

    def create_csnet(self, data, feature="color", training=True):

        with tf.name_scope('conv_1') as scope:
            conv1 = create_conv2d(data, self.W_conv1)

        with tf.name_scope('relu_1') as scope:
            relu1 = create_relu(conv1, self.b_conv1)

        with tf.name_scope('pool_1') as scope:
            pool1 = create_max_pool(relu1)

        with tf.name_scope('conv_2') as scope:
            conv2 = create_conv2d(pool1, self.W_conv2)

        with tf.name_scope('relu_2') as scope:
            relu2 = create_relu(conv2, self.b_conv2)

        with tf.name_scope('pool_2') as scope:
            pool2 = create_max_pool(relu2)

        with tf.name_scope('fc_1') as scope:
            pool_shape = pool2.shape
            reshape = tf.reshape(pool2, [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
            fc1 = tf.nn.relu(tf.matmul(reshape, self.W_fc1) + self.b_fc1)

        fc1 = tf.layers.dropout(fc1, 0.5, training=training)

        with tf.name_scope('fc_2') as scope:
            fc2 = tf.matmul(fc1, self.W_fc2) + self.b_fc2

        with tf.name_scope("mask") as scope:
            m_feature = tf.cond(tf.equal(feature, tf.constant("color")), lambda: self.M[:, 0], lambda: self.M[:, 1])
            m_feature = tf.nn.relu(m_feature)
            out = tf.multiply(fc2, m_feature)

        return out