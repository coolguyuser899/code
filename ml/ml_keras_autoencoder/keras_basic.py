# from keras.models import Sequential
# from keras.layers import Dense, Activation
#
# model = Sequential([
#     Dense(5, input_shape=(3,), activation='relu'),
#     Dense(2, activation='softmax'),
# ])
#
# # print(model)
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# image = mpimg.imread("NN.png")
# plt.imshow(image)
# plt.show()

#activation function, defines output giving input, output = activation(weighted sum of inputs)
# from keras.models import Sequential
# from keras.layers import Dense, Activation
#
# model = Sequential ([
#     Dense(5, input_shape=(3,), activation='relu')
# ])
#
# ##same as the following code
# model = Sequential()
# model.add(Dense(5, input_shape=3,))
# model.add(Activation('relu'))

#Training a neural network
#SGD stochastic gradient descent, to minimize the loss (incremental gradient descent)

#Learing process of neural network
# epoch - one pass of data
# learning rate: between 0.1 and 0.001
# d(loss) / d(weight) x 0.001
# Adam = variation of SDG for optimization

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#
# #build a model, input layer 1, 2 hidden layer
# model = Sequential([
#     Dense(16, input_shape=(1,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='softmax')
# ])
#
# #Adam optimizer, learning rate = 0.0001, loss = type, matrics = what is to print out
# model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# for a single-input model with 2 classes (binary):

#### keras example
# model = Sequential()
# model.add(Dense(1, input_dim=784, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # generate dummy data
# import numpy as np
# scaled_train_samples = np.random.random((1000, 784))
# train_labels = np.random.randint(2, size=(1000, 1))
#
# #np array training data, labels, how many piece data feed at once, how many passes
# model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)

####loss function
# minimize lost with continuous update weights and model during training
# MSE mean squared error


##Different dataset used in train, test, validation
#train data set, used to compute again and again
#overfitting, model is good at training data predication, but not good at validation data set
#testset, different from train and validation data, has to be unlabeled, without knowing what it is

# import numpy as np
# scaled_train_samples = np.random.random_sample((10,))
# train_labels = np.random.randint(2, size=(10, 1))
# print(scaled_train_samples, train_labels)
#
# model = Sequential([
#     Dense(16, input_shape=(1,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='softmax')
# ])
#
# #Adam optimizer, learning rate = 0.0001, loss = type, matrics = what is to print out
# model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #use 20% of training set as validation set
# model.fit(scaled_train_samples, train_labels, validation_split=0.20, batch_size=10, epochs=20, shuffle=True, verbose=2)

# ##or use separate set
# valid_set = [(sample, label),(sample, label),....,(sample, label),(sample, label)]
# model.fit(scaled_train_samples, train_labels, validation_data=valid_set, batch_size=10, epochs=20, shuffle=True, verbose=2)


##overfitting, model is good at classfifying and predicting training set, but not good at data which is trained on or test set
# if validation accuracy is worse then training, overfitting indicator
# model is not generalized
# solution, 1) add more data to training set
# 2) data augmentation (rotation, scaling, flip),
# 3) reduce complexity of model
# 4) dropout of nodes at certain layer

##underfitting
# model is not good at data which is trained on
# solution: 1) add complexity to model, model is too simple, increase neurons or layers
# 2) add more features to the training set
# 3) reduce dropout, define percentage of dropout

### Regularization in neural network, reduce overfitting or reduce variant by not penalizing complexity
# L2 regularization loss + x
# from keras import regularizers
# model = Sequential([
#     Dense(16, input_shape=(1,), activation='relu'),
#     Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#     Dense(16, input_shape=(1,), activation='sigmoid'),
# ])

#### learning rate, often between 0.01 to 0.0001
# model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentrypy', metrics=['accuracy'])
# model.optimizer.lr = 0.01
# model.optimizer.lr


#### Batch size number of sample passed to network (as a group)
# value must be tested and tuned for better result
#batch_size=10   #pass data set 10 at a time

#### Fine tuning a neural network or transfer learning
#use exisiing model by remove last layer or remove/add more hidden layer for mew model

#### Data augmentation, create new data based on existing data, horizontal flip
#rotate, shift width, shift hight, zooming, varying color, flip


####predicating - have model to predict
#pass un-labelled data to match model on test data
# predictions = model.predict(scaled_test_sample, batch_size=10, verbose=0)
# for i in predictions:
#     print(i)


####Supervised learning - training and validation data are labeled
#data (sample, label) -> output, they are used to learn data
#
# import keras
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Activation
# from keras.layers.core import Dense
# from keras.optimizers import Adam
#
# model = Sequential([
#     Dense(16, input_shape=(2,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='sigmoid')
# ])
#
# model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #weight, height
# train_samples = [[150, 67], [130, 60], [200, 65], [125, 52], [230, 72], [181, 70]]
# # 0: male
# # 1: female
# train_labels = [1, 1, 0, 1, 0, 0]
# model.fit(x=train_samples, y=train_labels, batch_size=3, epochs=10, shuffle=True, verbose=2)

####Unsupervised learning
#training data is not labeled, no way to measure the accuracy
#one popular is clusterign algorithm, create learning structure
#autoencoder - handwritten number as input and reconstruct image
#original input -> encoder -> compressed representation-> reconstructed input
#why we need unsupervised learing and reconstructing image
#denoise image, and reconstruct better quality image


####Semi-supervised learning using both labeled and unlabled data
#use pseudo-labeling
#add label in predict process


####Convolutional neural network (CNNs)
#so far the most popular network for analyzing image, but it can also used ofr data classification as well
#convolutional layer use filters to detect pattern
#MLP multi-layer percepron
#hidden layer is convolutional layers
#filter slide over each 3x3 set of pixels from the input itself until it slid over every 3x3 block
#this sliding is referred to as convolving



