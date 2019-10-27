"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record <PATH_TO_ANNOTATIONS_FOLDER>/label_map.pbtxt

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record  --label_map <PATH_TO_ANNOTATIONS_FOLDER>/label_map.pbtxt
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys



def main(_):
    filenames = "C:\\Users\\wangx\\object_detection_demo\\data\\annotations\\train.record"
    raw_dataset = tf.data.TFRecordDataset(filenames)

    for raw_record in raw_dataset.take(10):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    # Create a description of the features.
    feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset

    load_pb("C:\\Users\\wangx\\object_detection_demo\\frozen_inference_graph.pb")

    def load_pb(path_to_pb):
        print("load_pb")
        with tf.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        input = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('output:0')

        return graph

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

