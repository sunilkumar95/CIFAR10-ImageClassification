from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import numpy as np
import tensorflow as tf


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """

  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name) as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  """Read the data and perform shallow whitening.

  Args:
    data_path: path to the data folder.
    num_valids: number of images reserved from the training images to use as
      validation data.

  Returns:
    images, labels: two dicts, each with keys ['train', 'valid', 'test']. They
      contain the images and labels for the corresponding category.
  """

  print("-" * 80)
  print("Reading data")

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  images["valid"] = images["train"][-num_valids:]
  labels["valid"] = labels["train"][-num_valids:]

  images["train"] = images["train"][:-num_valids]
  labels["train"] = labels["train"][:-num_valids]

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print("Preprocess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels


def create_batches(images, labels, batch_size=32, eval_batch_size=100):
  """Creates the TF queues and samples [images] and [labels].

  Args:
    images, labels: outputs of the function [read_data] above.

  Returns:
    batched_images: a dict with ['train', 'valid', 'test'], each is a TF tensor
      of size [N, H, W, C], representing a minibatch of images.
    batched_labels: same with batched_images, but each dict entry is a TF tensor
      of size [N], representing a minibatch of corresponding labels.
  """

  with tf.device("/cpu:0"):
    # training data
    num_train_examples = np.shape(images["train"])[0]
    num_train_batches = (
      num_train_examples + batch_size - 1) // batch_size
    x_train, y_train = tf.train.shuffle_batch(
      [images["train"], labels["train"]],
      batch_size=batch_size,
      capacity=5000,
      enqueue_many=True,
      min_after_dequeue=0,
      num_threads=16,
      allow_smaller_final_batch=True,
    )

    def _pre_process(x):
      x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
      x = tf.random_crop(x, [32, 32, 3])
      x = tf.image.random_flip_left_right(x)

      return x
    x_train = tf.map_fn(_pre_process, x_train, back_prop=False)

    # valid data
    num_valid_examples = np.shape(images["valid"])[0]
    num_valid_batches = (
      (num_valid_examples + eval_batch_size - 1) // eval_batch_size)
    x_valid, y_valid = tf.train.batch(
      [images["valid"], labels["valid"]],
      batch_size=eval_batch_size,
      capacity=500,
      enqueue_many=True,
      num_threads=1,
      allow_smaller_final_batch=True,
    )

    # test data
    num_test_examples = np.shape(images["test"])[0]
    num_test_batches = (
      (num_test_examples + eval_batch_size - 1) // eval_batch_size)
    x_test, y_test = tf.train.batch(
      [images["test"], labels["test"]],
      batch_size=eval_batch_size,
      capacity=1000,
      enqueue_many=True,
      num_threads=1,
      allow_smaller_final_batch=True,
    )

  batched_images = {
    "train": x_train,
    "valid": x_valid,
    "test": x_test,
  }
  batched_labels = {
    "train": y_train,
    "valid": y_valid,
    "test": y_test,
  }
  return batched_images, batched_labels

