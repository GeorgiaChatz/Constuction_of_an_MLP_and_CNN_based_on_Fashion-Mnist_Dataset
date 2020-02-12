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


NUM_EPOCHS = 30
BS = 64
INIT_LR=1e-2
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
l2=regularizers.l2(0.01)


fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print(' - X_train.shape = {}, Y_train.shape = {}'.format(X_train.shape, Y_train.shape))
print(' - X_test.shape = {}, Y_test.shape = {}'.format(X_test.shape, Y_test.shape))

# keep an non pre-processed copy of X_test/y_test for visualization
test_images, test_labels = X_test.copy(), Y_test.copy()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=13)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_val = X_val.astype('float32')/ 255.0


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))


print('After preprocessing:')
print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, Y_train.shape))
print(' - X_val.shape = {}, y_val.shape = {}'.format(X_val.shape, Y_val.shape))
print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, Y_test.shape))
print(' - test_images.shape = {}, test_labels.shape = {}'.format(test_images.shape,
test_labels.shape))


model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=l2,
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2 ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.20),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=l2 ),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=l2),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),

        Flatten(),

        Dense(1024, activation='relu',kernel_regularizer=l2),
        Dropout(0.40),

        Dense(512, activation='relu',kernel_regularizer=l2),
        Dropout(0.30),

        Dense(NUM_CLASSES, activation='softmax')
    ])



#Data Augmentation 
#datagen = ImageDataGenerator(
        #featurewise_center=False,  # set input mean to 0 over the dataset
        #samplewise_center=False,  # set each sample mean to 0
        #featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #samplewise_std_normalization=False,  # divide each input by its std
        #zca_whitening=False,  # apply ZCA whitening
        #rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        #zoom_range = 0.01, # Randomly zoom image
        #width_shift_range=0.03,  # randomly shift images horizontally (fraction of total width)
        #height_shift_range=0.03,  # randomly shift images vertically (fraction of total height)
        #horizontal_flip=False,  # randomly flip images
        #vertical_flip=False)  # randomly flip images

adam = Adam(lr=0.0001, decay=1e-6)
# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model.compile(optimizer=adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=NUM_EPOCHS,batch_size=BS, validation_data=(X_val, Y_val))
#model.fit_generator(datagen.flow(X_train,Y_train, batch_size=BS),
                              #epochs = NUM_EPOCHS, validation_data = (X_val,Y_val),
                              #verbose = 2, steps_per_epoch=X_train.shape[0] // BS)
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
