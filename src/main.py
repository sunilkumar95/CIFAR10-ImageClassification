from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import count_model_params
from src.utils import print_user_flags

from src.data_utils import read_data

from src.models import Hparams
from src.models import conv_net
from src.models import feed_forward_net

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", None, "Path to CIFAR-10 data")
DEFINE_string("output_dir", "output", "Path to log folder")

DEFINE_string("model_name", "",
              "Name of the method. [feed_forward|conv]")

DEFINE_integer("log_every", 10, "How many steps to log")
DEFINE_integer("num_classes", 10, "Number of classes")
DEFINE_integer("train_batch_size", 256, "Size of training batches")
DEFINE_integer("eval_batch_size", 100, "Size of testing batches")
DEFINE_float("l2_reg", 1e-4, "L2 regularization rate")
DEFINE_float("learning_rate", 0.05, "Learning rate")
DEFINE_integer("train_steps", 6000, "Number of training steps")

def get_ops(images, labels):
  """Builds the model."""

  print("-" * 80)
  print("Creating a '{0}' model".format(FLAGS.model_name))
  if FLAGS.model_name == "feed_forward":
    ops = feed_forward_net(images, labels, FLAGS)
  elif FLAGS.model_name == "conv":
    ops = conv_net(images, labels, FLAGS)
  else:
    raise ValueError("Unknown model name '{0}'".format(FLAGS.model_name))

  assert "global_step" in ops
  assert "train_op" in ops
  assert "train_loss" in ops
  assert "valid_acc" in ops
  assert "test_acc" in ops

  return ops


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {0} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {0} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print_user_flags()

  hparams = Hparams()
  images, labels = read_data(FLAGS.data_path)

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)

    # count model variables
    tf_variables = tf.trainable_variables()
    num_params = count_model_params(tf_variables)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=FLAGS.output_dir) as sess:

        # training loop
        print("-" * 80)
        print("Starting training")
        for step in range(1, hparams.train_steps + 1):
          sess.run(ops["train_op"])
          if step % FLAGS.log_every == 0:
            global_step, train_loss, valid_acc = sess.run([
              ops["global_step"],
              ops["train_loss"],
              ops["valid_acc"],
            ])
            log_string = ""
            log_string += "step={0:<6d}".format(step)
            log_string += " loss={0:<5.2f}".format(train_loss)
            log_string += " val_acc={0:<3d}/{1:<3d}".format(
              valid_acc, hparams.eval_batch_size)
            print(log_string)
            sys.stdout.flush()

        # final test
        print("-" * 80)
        print("Training done. Eval on TEST set")
        num_corrects = 0
        for _ in range(10000 // hparams.eval_batch_size):
          num_corrects += sess.run(ops["test_acc"])
        print("test_accuracy: {0:>5d}/10000".format(num_corrects))


if __name__ == "__main__":
  tf.app.run()
