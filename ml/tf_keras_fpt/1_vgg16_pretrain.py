"""
code from below link
https://github.com/sankit1/cv-tricks.com/blob/master/Tensorflow-tutorials/Keras-Tensorflow-Finetuning-tutorial/1_vgg16_pretrain.py

how to run:
python3.6 1_vgg16_pretrain.py -train ../tf_image_classifier/training_data/ -val ../tf_image_classifier/testing_data/ -num_class 2

output ....
Epoch 1/20
 99/100 [============================>.] - ETA: 7s - loss: 0.5368 - acc: 0.7121
Epoch 00001: val_loss improved from inf to 0.33226, saving model to pretrained_model.h5
......
Epoch 19/20
 54/100 [===============>..............] - ETA: 6:29 - loss: 0.1894 - acc: 0.9222
 99/100 [============================>.] - ETA: 8s - loss: 0.1969 - acc: 0.9182
Epoch 00019: val_loss did not improve
100/100 [==============================] - 1134s 11s/step - loss: 0.1952 - acc: 0.9190 - val_loss: 0.2424 - val_acc: 0.9000
Epoch 20/20
 17/100 [====>.........................] - ETA: 10:27 - loss: 0.1978 - acc: 0.9235
 99/100 [============================>.] - ETA: 7s - loss: 0.1912 - acc: 0.9182
Epoch 00020: val_loss did not improve
100/100 [==============================] - 1105s 11s/step - loss: 0.1906 - acc: 0.9190 - val_loss: 0.4050 - val_acc: 0.8400

"""

#from network import VGG16   #deprecated
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
import argparse
from time import time
import sys


#### 1 path to training and testing data
img_size = 224
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--train_dir", type=str, required=True, help="(required) the train data directory")
ap.add_argument("-val", "--val_dir", type=str, required=True, help="(required) the validation data directory")
ap.add_argument("-num_class", "--class", type=int, required=True, help="(required) number of clases to be trained")
args = vars(ap.parse_args())


#### 2 random image transformations
batch_size = 10

train_datagen = image.ImageDataGenerator(
    width_shift_range= 0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    args["train_dir"],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    args["val_dir"],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

#### 3 create vgg16 network graph, load pretrained weights
print('loading the model and the pre-trained weights...')

base_model = VGG16(include_top=False, weights='imagenet')
i = 0
for layer in base_model.layers:
    layer.trainable = False
    i += 1
    print(i, layer.name)

# sys.exit()


""" output from print()
Found 1000 images belonging to 2 classes.
Found 400 images belonging to 2 classes.
loading the model and the pre-trained weights...

1 input_1
2 block1_conv1
3 block1_conv2
4 block1_pool
5 block2_conv1
6 block2_conv2
7 block2_pool
8 block3_conv1
9 block3_conv2
10 block3_conv3
11 block3_pool
12 block4_conv1
13 block4_conv2
14 block4_conv3
15 block4_pool
16 block5_conv1
17 block5_conv2
18 block5_conv3
19 block5_pool

"""
#### 4 add the top as per number of classes in our dataset, use dropout layer with 0.2, discarding 20% weights
x = base_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(args["class"], activation='softmax')(x)

#### 5 specify complete model input, output, optimizer, loss
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
filepath = 'pretrained_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1 )
callbacks_list = [checkpoint, tensorboard]

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])
# model.compile(loss="categorical_crossentrypy", optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-8, decay=0.0), metrics=["accuracy"])

num_training_img = 1000
num_validation_img = 400
stepsPerEpoch = num_training_img / batch_size
validationSteps = num_validation_img / batch_size

model.fit_generator(
    train_generator,
    steps_per_epoch = stepsPerEpoch,
    epochs= 20,
    callbacks= callbacks_list,
    validation_data= validation_generator,
    validation_steps=validationSteps
)