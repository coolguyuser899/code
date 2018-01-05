"""
code is from the following link - Building Autoencoders in Keras
https://blog.keras.io/building-autoencoders-in-keras.html

"""

#### Version 1 Build basic simple autoencoder, run from intellij

from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 32 # 32 floats, compression of factor 24.5, input is 784 floats
input_img = Input(shape=(784,))   #input placeholder
encoded = Dense(encoding_dim, activation='relu')(input_img)  #encoded representation of the input
decoded = Dense(784, activation='sigmoid')(encoded)   #lossy reconstruction of the input

autoencoder = Model(input_img, decoded) #model maps input to its reconstruction

encoder = Model(input_img, encoded) #maps input to its encoded representation

encoded_input = Input(shape=(encoding_dim,)) #placeholder for an encoded 32-dimensional input
decoder_layer = autoencoder.layers[-1]  #retrieve the last layer of the autoencoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))    #create the decoder model

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')   #model use per-pixel binary crossentropy loss, and Adadelta optimizer

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()   #use mnist digits, discard labels

x_train = x_train.astype('float32') / 255.  #normalize values between 0 and 1, flatten the 28x28 images into vectors of size 784
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=30,      # basic approach
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test)
                )

#after 50 epochs, stable train/test loss valud of about 0.11, let's visualize the reconstructed input and the encoded representation

encoded_img = encoder.predict(x_test)   #taken digits from test set
decoded_img = decoder.predict(encoded_img)

import matplotlib.pyplot as plt

n = 10  #how many digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n, i +1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()  # result with basic approach

####Version 1 completed
















