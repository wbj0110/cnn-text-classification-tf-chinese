#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
import csv
import pickle
import data_helpers as dp

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "../runs/1508205193/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

runs_path = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, os.path.pardir))
vacabulary_path = os.path.join(runs_path, "vocab")
print("vacabulary path:"+vacabulary_path)
vacabulary_file = open(vacabulary_path, "rb")
vacabulary = pickle.load(vacabulary_file)
vacabulary_file.close()

#load sequence length
sequence_path = os.path.join(runs_path, "sequence_lenth")
sequence_file = open(sequence_path, "rb")
sequence_length = pickle.load(sequence_file)
sequence_file.close()

print("sequence is {0}",sequence_length)


def classify(text):
    x_text = [list(text.strip())]
    sentences_padded = dp.pad_sentences(x_text, sequence_length=sequence_length)
    x = np.array([[vacabulary.get(word,0) for word in sentence] for sentence in sentences_padded])
    print("\npredict...\n")
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            single_predictions = sess.run(predictions, {input_x: x, dropout_keep_prob: 1.0})
            print(single_predictions)
            return single_predictions