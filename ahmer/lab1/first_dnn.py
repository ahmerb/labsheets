############################################################
#                                                          #
#  Code for Lab 1: Your First Fully Connected Layer  #
#                                                          #
############################################################


import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd
from random import shuffle

def zip_with(xs, ys, f):
  return [ f(x, y) for (x, y) in zip(xs, ys) ]

sess = tf.Session()

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])

# gen random seed
np.random.seed(0)

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# separate data into inputs (x) and outputs (y)
all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

all_y = pd.get_dummies(data.iris_class)

n_x = len(all_x.columns)
n_y = len(all_y.columns)

# split data into training and testing (100 train, 50 test)
train_x = all_x[:100]
train_y = all_y[:100]
test_x  = all_x[100:150]
test_y  = all_y[100:150]

# sanity check
print(all_x.shape, all_y.shape)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# 3.2 define a perceptron

# x and y can be any number of rows, but n_x=4 and n_y=3 columns
x = tf.placeholder(tf.float32, shape=[None, n_x])
y = tf.placeholder(tf.float32, shape=[None, n_y])

# define W and b, but using random/zero weights
W = tf.get_variable("W", dtype=tf.float32, shape=[n_x, n_y], initializer=tf.zeros_initializer)
b = tf.get_variable("b", dtype=tf.float32, shape=[n_y], initializer=tf.zeros_initializer)

#  do y = Wx-b to get predication for y using current zero init'd weights
prediction = tf.nn.softmax(tf.matmul(x, W) - b)

# 3.3 train a perceptron

# define cost fn (defines difference between prediction and ground truth)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))

# define optimizer to actually do training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# run optimizer
sess.run(tf.global_variables_initializer())

for epoch in range(10000):
  sess.run([optimizer], feed_dict={ x: train_x, y: train_y })
  if epoch % 100 == 0:
    # results is an array [prob_of_1, prob_of_2, prob_of_3] with the results on test_x
    results = sess.run(prediction, feed_dict={ x: test_x[:1], y: test_y[:1] }).tolist()
    accuracy = calc_accuracy(results, train_y, 100)
    print("Accuracy of Perceptron at epoch", epoch, "is", accuracy)

print("===Training done===")

def calc_accuracy(results, ground_truths, n):
  findMaxIx = lambda result: np.argmax(result)
  result_classes = np.vectorize(findMaxIx)(results)
  f = lambda x, y: 1 if x == y else 0
  return reduce(lambda x, y: x + y, zip_with(result_classes, ground_truths, f) / n


# 3.4 test your tained perceptron
# also add line to training to print accuracy at every 100 epochs

sess.run(prediction, feed_dict={ x: test_x[:1], y: test_y[:1] }).toList()[0]

