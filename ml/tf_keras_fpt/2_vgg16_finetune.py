"""
code from below link
https://github.com/sankit1/cv-tricks.com/blob/master/Tensorflow-tutorials/Keras-Tensorflow-Finetuning-tutorial/1_vgg16_pretrain.py

how to run:
python3.6 2_vgg16_finetune.py  -train ../tf_image_classifier/training_data/ -val ../tf_image_classifier/testing_data/ -num_class 2

output with 98% accuracy

Epoch 1/20
  6/100 [>.............................] - ETA: 14:04 - loss: 0.0737 - acc: 1.0000


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

#### 4 add the top as per number of classes in our dataset, use dropout layer with 0.2, discarding 20% weights
x = base_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(args["class"], activation='softmax')(x)

#### 5 output file name
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
filepath = 'fine_tuned_model.h5'

#### used in pretrain
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1 )
#### used in finetune
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1 )

callbacks_list = [checkpoint, tensorboard]

model = Model(inputs=base_model.input, outputs=predictions)

######### load pretrained model and fine tune
model.load_weights("pretrained_model.h5")
model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])
#########

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