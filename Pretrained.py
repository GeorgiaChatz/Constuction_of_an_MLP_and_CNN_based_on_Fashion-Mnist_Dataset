#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,BatchNormalization,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import keras.callbacks
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
import cv2
from keras.preprocessing.image import img_to_array, array_to_img

NUM_EPOCHS = 50
BS = 86
INIT_LR=1e-2
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
l2=regularizers.l2(0.01)


fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print(' - X_train.shape = {}, Y_train.shape = {}'.format(X_train.shape, Y_train.shape))
print(' - X_test.shape = {}, Y_test.shape = {}'.format(X_test.shape, Y_test.shape))


# keep an non pre-processed copy of X_test/y_test for visualization
test_images, test_labels = X_test.copy(), Y_test.copy()

X_train = [cv2.cvtColor(cv2.resize(i, (56,56)), cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.cvtColor(cv2.resize(i, (56,56)), cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# convert to one-hot-encoding(one hot vectors)
Y_train = to_categorical(Y_train, num_classes = 10)
# convert to one-hot-encoding(one hot vectors)
Y_test = to_categorical(Y_test, num_classes = 10)


print('After preprocessing:')
print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, Y_train.shape))
print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, Y_test.shape))
print(' - test_images.shape = {}, test_labels.shape = {}'.format(test_images.shape,
test_labels.shape))

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=13)

conv_base = keras.applications.vgg16.VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(56, 56, 3)
                 )

model = models.Sequential()
model.add(conv_base) # pretrain model
model.add(layers.Flatten())
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#leakyrelu


adam = Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(X_train, Y_train,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_val, Y_val))
print("Evaluating... ")
print('Training data:', flush=True)
loss, acc = model.evaluate(X_train, Y_train, verbose=1)
print("  Training : loss %.3f - acc %.3f" % (loss, acc))
print('Cross-validation data:', flush=True)
loss, acc = model.evaluate(X_val, Y_val, verbose=1)
print("  Cross-val: loss %.3f - acc %.3f" % (loss, acc))
print('Test data:', flush=True)
loss, acc = model.evaluate(X_test, Y_test, verbose=1)
print("  Testing  : loss %.3f - acc %.3f" % (loss, acc))

