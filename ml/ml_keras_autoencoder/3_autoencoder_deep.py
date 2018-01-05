"""
Version add deep learning or more layers

ouptut
Epoch 1/100
  256/60000 [..............................] - ETA: 2:50 - loss: 0.6929
  60000/60000 [==============================] - 5s 76us/step - loss: 0.3486 - val_loss: 0.2651
  ......
59904/60000 [============================>.] - ETA: 0s - loss: 0.0970
60000/60000 [==============================] - 4s 66us/step - loss: 0.0970 - val_loss: 0.0958
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_dim = 32  # change to 32 to match line 27 and 29
input_img = Input(shape=(784,))     # 32 floats - compression of factor 24.5, assuming the input is 784 floats

#### add sparsity contraint
# encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
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
                epochs=100,               #epochs in plural form
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
