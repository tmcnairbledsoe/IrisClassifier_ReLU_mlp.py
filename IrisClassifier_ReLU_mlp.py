import tensorflow as tf# Common imports
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Set seed for shuffling
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# To make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

# Get the iris data set
iris   = datasets.load_iris()
data   = iris["data"]
target = iris["target"]

#Set the layers size
n_inputs = 4 # 4 inputs, 1 for each petal length
n_hidden1 = 256 # Number of hidden layers
n_outputs = 3 # Number of outputs, 1 for each flower type

# Set the place holders 
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# function for creating neuron layers
# Could be replaced with dense but will keep this for reference
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.ones([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
        
#Create our DNN using the function above or the dense function
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden1, n_outputs, name="outputs")
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.25, random_state=RANDOM_SEED)
train_y

#Execution phase and output
n_epochs = 40
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in range(len(train_X)):
            sess.run(training_op, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
            
        acc_train = accuracy.eval(feed_dict={X: train_X, y: train_y})
        acc_test = accuracy.eval(feed_dict={X: test_X, y: test_y})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
