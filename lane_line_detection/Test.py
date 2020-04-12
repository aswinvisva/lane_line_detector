import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow_core.python.keras.layers import BatchNormalization, Conv2DTranspose, Conv2D
import cv2

def generator():
    (X_Train, y_train), (X_Test, y_test) = tf.keras.datasets.mnist.load_data()

    X_Train=X_Train/255

    i = 0
    while True:
        if i >= 50000:
            break
        i=i+1
        yield (np.array(X_Train[i]).reshape((1,28,28,1)), np.array(X_Train[i]).reshape((1,28,28,1)))

(X_Train, y_train), (X_Test, y_test) = tf.keras.datasets.mnist.load_data()
X_Train = X_Train.reshape(X_Train.shape[0], 28, 28, 1)
X_Test = X_Test.reshape(X_Test.shape[0], 28, 28, 1)
X_Train=X_Train/255
X_Test=X_Test/255
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
input = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(49, activation="softmax")(x)
x = tf.keras.layers.Reshape((7, 7, 1))(x)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

model = tf.keras.Model(input, x)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x=X_Train, y=X_Train, batch_size=32, epochs=10)

predictions = model.evaluate(X_Test,X_Test)
predictions = model.predict(X_Test)

image = predictions[0]
image=image*255

while True:
    cv2.imshow("prediction", image)
    cv2.imshow("image", X_Test[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(predictions)

