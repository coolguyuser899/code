#### Version 2 - from PCA (principal component analysis) to aparsity contraint on hidden layer, so fewer unit would "fire" at a given time
# add this - activity_regularizer=regularizers.l1(10e-5)
# model ends with training loss of 0.11, and test loss of 0.10, due to regularization added to the loss during training

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_dim = 32  # encoded representation
input_img = Input(shape=(784,))     # 32 floats - compression of factor 24.5, assuming the input is 784 floats

#### add sparsity contraint
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded) # map input to reconstruction

encoder = Model(input_img, encoded) # encoder model maps an input to encoded represendation

encoded_input = Input(shape=(encoding_dim,))    #create decoder model, create placeholder for an encoded (32-dim) input
decoder_layer = autoencoder.layers[-1]     # retrieve the last layer of the autoencoder layer
decoder = Model(encoded_input, decoder_layer(encoded_input))    # create the decoder model

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')   #configure mode

from keras.datasets import mnist    #load input data using MNIST digits
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()   #discard second value

#normalize all values between 0 and 1, flatten 28x28 images into vectors of size 784
x_train = x_train.astype('float32') / 255.  # copy array cast to float32 type, Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
x_test = x_test.astype('float32') / 255.    # 255. period = 255.0, a float
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))   #shape return product, reshape return new matrix
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)    #shape return array matrix
print(x_test.shape)

autoencoder.fit(x_train, x_train,       #train data in 100 epochs, see stable train/test loss value of about 0.11
                epochs=2,               #epochs in plural form
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


#visualize reconstructed input and encoded representations
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)     #display original
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1 +n)  # display reconstruction
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
