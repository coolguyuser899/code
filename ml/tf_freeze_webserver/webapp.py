"""
tf freezing is the process to identify and save all of required things (graph, weithgts, etc) in a single file that can be used

tf model contains 4 files
model-ckpt.meta - complete graph
model-ckpt.data-0000-of-000001: values of variables (weights, biases, placeholders, gradients, hyper-parameters
model-ckpt.index: metadata of a tensor
checkpoint: checkpoint information

single encapsulated file(.pb) - frozen graph def, a serialized graph_def protocol buffer written to disk, freezed trained model

#to run
python3 webapp.py

"""

import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import jsonify
import base64
# import StringIO  #deprecated
from io import StringIO
import tensorflow as tf
import numpy as np
import cv2

#obtain flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER = 'images'
def load_graph(trained_model):
    with tf.gfile.GFile(trained_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map = None,
            return_elements= None,
            name=""
        )
    return graph

@app.route('/')
def index():
    return "web server is running"

@app.route('/demo', methods=['POST', 'GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_size = 128
        num_channels = 3
        images = []

        image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        # images.append(image)
        # images = np.array(images, dtype=np.unit8)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)

        #
        x_batch = images.reshape(1, image_size, image_size, num_channels)
        graph = app.graph
        y_pred = graph.get_tensor_by_name("y_pred:0")
        #feed image to input placehodler
        x = graph.get_tensor_by_name("x:0")
        y_test_images = np.zeros((1,2))
        sess = tf.Session(graph=graph)

        feed_dict_testing = {x: x_batch}
        result = sess.run(y_pred, feed_dict = feed_dict_testing)
        # [probability of cats, probability of dogs]
        #print()
        pred = str(result[0][0]).split(" ")
        #print(pred)


        out = {"cat": str(result[0][0]), "dogs": str(result[0][1])}
        return jsonify(out)
        ##return redirect(url_for('just_upload', pic = filename))

    # return """
    #     demo
    # """
    return  """
    <!doctype html>
    <html lang="en">
    <head>
        <title>Running my first AI demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" > Home </a>
            </nav>
            <div class="inner cover">

            </div>
            <div class="mastfoot">
                <hr />
                <div class="container">
                    <div style="margin-top:5%">
                        <h1 style="color:black">Dogs Cats Classfification Demo</h1>
                        <h4 style="color:black">Upload New Image </h4>
                        <form method=post enctype=multipart/form-data>
                            <p><input type=file name=file>
                            <input type=submit style="color:black;" value=Upload>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    </body>
    </html>

    """

app.graph = load_graph('./dogs-cats-model.pb')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port = int("5000"), debug=True, use_reloader = False)