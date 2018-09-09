import os
import sys

import numpy as np
import tensorflow as tf

from src.utils import count_model_params

from src.data_utils import create_batches

class Hparams:
  # number of output classes. Must be 10 for CIFAR-10
  num_classes = 10

  # size of each train mini-batch
  batch_size = 250

  # size of each eval mini-batch
  eval_batch_size = 100

  learning_rate = 0.05
  # l2 regularization rate
  l2_reg = 1e-4

  # number of training steps
  train_steps = 6000


def conv_net(images, labels, *args, **kwargs):
  """A conv net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """

  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  N, H, W, C = (x_train.get_shape()[0].value,
                x_train.get_shape()[1].value,
                x_train.get_shape()[2].value,
                x_train.get_shape()[3].value)

  # create model parameters
  def _get_logits(x, kernels, channels, l, flag, p, num_classes=hparams.num_classes):
    # x, mean, var = tf.nn.fused_batch_norm(x, scale=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], is_training=flag)
    layers = list(zip(kernels, channels))
    
    # 2 Convolutional Layers
    for layer, (k, c) in enumerate(layers[:l]):
      with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
        currc = x.get_shape()[-1].value
        w = tf.get_variable("w", [k, k, currc, c])
      x = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME")
      x = tf.nn.relu(x)
    
    # Pooling Layer
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 2 Convolutional Layers
    for layer, (k, c) in enumerate(layers[l:]):
      with tf.variable_scope("layer_{}".format(layer+2), reuse=tf.AUTO_REUSE):
        currc = x.get_shape()[-1].value
        w = tf.get_variable("w", [k, k, currc, c])
      x = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME")
      x = tf.nn.relu(x)

    # Pooling Layer
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # Feed Forward Layer
    H, W, C = (x.get_shape()[1].value,
               x.get_shape()[2].value,
               x.get_shape()[3].value)
    x = tf.reshape(x, [-1, H*W*C])
    currdim = x.get_shape()[-1].value
    with tf.variable_scope("fflayer{}".format(1), reuse=tf.AUTO_REUSE):
      w = tf.get_variable('w', [currdim, currdim])
      x = tf.matmul(x, w)
      # Dropout Layer
      if flag:
        x = tf.nn.dropout(x, p)
      x = tf.nn.relu(x)

    #Logits Layer
    with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
      currc = x.get_shape()[-1].value
      w = tf.get_variable("w", [currc, num_classes])
    logits = tf.matmul(x, w)
    return logits

  kernels = [3, 5, 7, 9]
  channels = [128, 256, 128, 256]
  pdrop = 0.8
  l = 2
  train_logits = _get_logits(x_train, kernels, channels, l, flag=True, p=pdrop)
  valid_logits = _get_logits(x_valid, kernels, channels, l, flag=False, p=pdrop)
  test_logits = _get_logits(x_test, kernels, channels, l, flag=False, p=pdrop)

  # create train_op and global_step
  beta = hparams.l2_reg
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, 
                                                              labels=y_train)
  train_loss = tf.reduce_mean(log_probs)

  # Add l2 regularization
  regularizer = tf.losses.get_regularization_loss()
  train_loss = train_loss + beta*regularizer

  # Create the training op
  optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops


def feed_forward_net(images, labels, name='feed_forward_net', *args, **kwargs):
  """A feed_forward_net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """
  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  N, H, W, C = (x_train.get_shape()[0].value,
                x_train.get_shape()[1].value,
                x_train.get_shape()[2].value,
                x_train.get_shape()[3].value)

  # create model parameters
  def _get_logits(x, dims, flag, p, num_classes=hparams.num_classes):
    x = tf.reshape(x, [-1, H*W*C])
    for layer, dim in enumerate(dims):
      currdim = x.get_shape()[-1].value
      with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [currdim, dim])
      x = tf.matmul(x, w)
      x = tf.nn.relu(x)
      if flag:
        x = tf.nn.dropout(x, p)
    currdim = x.get_shape()[-1].value
    with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
      w = tf.get_variable("w", [currdim, num_classes])
      logits = tf.matmul(x, w)
    return logits


  dims = [256, 512, 128]
  pdrop = 0.8
  train_logits = _get_logits(x_train, dims=dims, flag=True, p = pdrop)
  valid_logits = _get_logits(x_valid, dims=dims, flag=False, p = pdrop)
  test_logits = _get_logits(x_test, dims=dims, flag=False, p = pdrop)

  # create train_op and global_step
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=train_logits, labels=y_train)
  train_loss = tf.reduce_mean(log_probs)
  optimizer = tf.train.AdagradOptimizer(
    learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops

