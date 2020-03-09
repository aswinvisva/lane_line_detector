import tensorflow as tf
import pandas as pd
import numpy as np

class Model:

    def __init__(self, latent_dim=50):

        self.CE = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.CD = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

        self.CAD = tf.keras.Sequential()
        self.CAD.add(self.CE)
        self.CAD.add(self.CD)
        self.CAD.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

        self.Flow = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(3,2), return_sequences=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation='relu')),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='relu')
        ])

        self.Flow.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

        print(self.Flow.summary())
        print(self.CD.summary())
        print(self.CE.summary())


    def train_CAD(self, train, labels):
        self.CAD.fit(train, labels, epochs=50, batch_size=128)

    def train_Flow(self, train, labels):
        self.Flow.fit(train, labels, epochs=50, batch_size=128)

if __name__ == '__main__':
    model = Model()


