# some borrowed from tensorflow example

from tensorflow.examples.tutorials.mnist import input_data
from data_utils import *
from convnet import *

# setting
DATA_SIZE = 5000
BATCH_SIZE = 100
NUM_ITERATIONS = 1000
VAL_INTERVAL = 100
DIM_EMBEDDING = 10
NUM_FEATURES = 2
SEED = 1


def compute_euclidean_distance(x, y):

    d = tf.square(tf.subtract(x, y))
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

    with tf.name_scope("triplet_loss"):
        d_p = compute_euclidean_distance(anchor_feature, positive_feature)
        d_n = compute_euclidean_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0., d_p - d_n + margin)
        return tf.reduce_mean(loss), tf.reduce_mean(d_p), tf.reduce_mean(d_n)


# set a seed
np.random.seed(SEED)

# import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = mnist.train.images[0:DATA_SIZE]
label_digit = np.argmax(mnist.train.labels[0:DATA_SIZE], axis=1)

# change a color randomly
data, label_color = change_color(data)
data = np.reshape(data, [-1, 28, 28, 3])

# make a data shuffler
data_shuffler = DataShuffler(data, label_color, label_digit)

# clear old variables
tf.reset_default_graph()

# setup input
train_anchor_data = tf.placeholder(tf.float32, [None, 28, 28, 3])
train_positive_data = tf.placeholder(tf.float32, [None, 28, 28, 3])
train_negative_data = tf.placeholder(tf.float32, [None, 28, 28, 3])
feature = tf.placeholder(tf.string)
training = tf.placeholder(tf.bool)

# make a model
convnet_architecture = ConvNet(dim_embedding=DIM_EMBEDDING, num_features=NUM_FEATURES)
convnet_train_anchor = convnet_architecture.create_convnet(train_anchor_data, feature)
convnet_train_positive = convnet_architecture.create_convnet(train_positive_data, feature)
convnet_train_negative = convnet_architecture.create_convnet(train_negative_data, feature)

loss, positives, negatives = compute_triplet_loss(convnet_train_anchor, convnet_train_positive, convnet_train_negative, 1)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    variables = [loss, positives, negatives, optimizer]

    for step in range(NUM_ITERATIONS):
        num = np.random.choice(2, 1)
        feat = "color" if num == 0 else "digit"
        batch_anchor, batch_positive, batch_negative = data_shuffler.get_triplet(n_triplets=BATCH_SIZE, feature=feat)
        print("minibatch constructed")
        feed_dict = {train_anchor_data: batch_anchor/256,
                     train_positive_data: batch_positive/256,
                     train_negative_data: batch_negative/256,
                     feature: feat,
                     training: True}
        loss, positives, negatives, _ = session.run(variables, feed_dict=feed_dict)
        print("Iteration {0} (feature: {4}): with minibatch training loss {1:.3g}, {2:.3g}, {3:.3g}".format(step, loss, positives, negatives, feat))
