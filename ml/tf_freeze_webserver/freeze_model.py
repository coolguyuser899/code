"""
python3.6 freeze_model.py

"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import os, sys

output_node_names = "y_pred"
saver = tf.train.import_meta_graph('dogs-cats-model.meta', clear_devices=True)
graph = tf.get_default_graph()

input_graph_def = graph.as_graph_def()

sess = tf.Session()
saver.restore(sess, "./dogs-cats-model")   ##this directory needs to be created
output_graph_def = graph_util.convert_variables_to_constants(
    sess,       #retrieve weights
    input_graph_def,  #retrieve node
    output_node_names.split(",") #select the usefull nodes
)

output_graph = "dogs-cats-model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()

