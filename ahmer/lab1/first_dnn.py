############################################################
#                                                          #
#  Code for Lab 1: 3.5 onwards (first deep fully connected network)  #
#                                                          #
############################################################

from __future__ import division # adopt python3 behaviour for division ('/' and '//')

import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd
from random import shuffle
from functools import reduce

def zip_with(xs, ys, f):
  return [ f(x, y) for (x, y) in zip(xs, ys) ]

def calc_accuracy(results, ground_truths, n):
  # map results ([[0.1,0.66,0.34],...]) -> selected class ([1,...])
  result_classes = np.fromiter((np.argmax(result) for result in results), dtype=float)

  # return percentage of result_classes that matches ground_truths
  f = lambda x, y: 1 if x == y else 0
  corrects = zip_with(result_classes, ground_truths, f)
  return reduce(lambda x, y: x + y, corrects) / n

# Use TensorFlow graph
# We will save summary with loss and accuracy for each batch
logs_path = './logs/'
g = tf.get_default_graph()
with g.as_default():

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

  # x and y can be any number of rows, but n_x=4 and n_y=3 columns
  x = tf.placeholder(tf.float32, shape=[None, n_x])
  y = tf.placeholder(tf.float32, shape=[None, n_y])

  # define W, b and h for the first layer
  h1 = 10
  W_fc1 = tf.Variable(tf.truncated_normal([n_x, h1], stddev=0.1))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[h1]))
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  # define layer 2
  h2 = 20
  W_fc2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[h2]))
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  # define layer 3
  h3 = 10
  W_fc3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.1))
  b_fc3 = tf.Variable(tf.constant(0.1, shape=[h3]))
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

  # define output layer
  W_fc4 = tf.Variable(tf.truncated_normal([h3, n_y], stddev=0.1))
  b_fc4 = tf.Variable(tf.constant(0.1, shape=[n_y]))
  prediction_fcn = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

  # define cost fn (defines difference between prediction and ground truth)
  with tf.name_scope('loss'): # add loss to summary
    cost_fcn = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction_fcn, scope="Cost_Function")
    tf.summary.scalar('loss', cost_fcn)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction_fcn, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

  # define optimizer to actually do training
  optimizer = tf.train.AdagradOptimizer(0.1).minimize(cost_fcn)

  # merge summaries
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(logs_path + '/train')
  test_writer  = tf.summary.FileWriter(logs_path + '/test')

  # run optimizer
  sess.run(tf.global_variables_initializer())

  # train and test every 100 epochs, writing to summaries
  for epoch in range(3000):
    train_summary, _ = sess.run([merged, optimizer], feed_dict={ x: train_x, y: train_y })

    # test performance every 100 epochs
    if epoch % 100 == 0:
      # results is an array [prob_of_1, prob_of_2, prob_of_3] with the results on test_x
      test_summary, results = sess.run([merged, prediction_fcn], feed_dict={ x: test_x, y: test_y })

      # add summaries to logs
      train_writer.add_summary(train_summary, epoch)
      test_writer.add_summary(test_summary, epoch)

      # monitor accuracy on test set
      ground_truths = np.fromiter((np.argmax(y) for y in test_y.get_values()), dtype=float)
      accuracy = calc_accuracy(results, ground_truths, 50) # 50 = len(results)
      print("Accuracy of my first dnn at epoch", epoch, "is", accuracy)


  print("===Training done===")
