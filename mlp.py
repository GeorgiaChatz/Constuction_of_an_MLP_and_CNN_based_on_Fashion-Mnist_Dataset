# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
			   

#plt.imshow(train_images[8],cmap=plt.cm.binary)
#plt.show()

train_images = train_images/255.0
test_images = test_images/255.0



model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(530, activation="relu"),
	keras.layers.Dropout(0.20),
	keras.layers.Dense(10, activation="softmax")
	])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=20, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested acc: ", test_acc)

prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[8])])  


