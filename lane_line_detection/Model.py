import ast
import os
import re

import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, \
    Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



class Model:

    def __init__(self, latent_dim=49):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        # ENCODER
        inp = Input((896,896, 1))
        e = Conv2D(32, (3, 3), activation='relu')(inp)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(64, (3, 3), activation='relu')(e)
        e = MaxPooling2D((2, 2))(e)
        e = Dropout(0.25)(e)
        e = Conv2D(64, (3, 3), activation='relu')(e)
        l = Flatten()(e)
        l = Dropout(0.5)(l)
        l = Dense(49, activation='softmax')(l)
        # DECODER
        d = Reshape((7, 7, 1))(l)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

        self.CAD = tf.keras.Model(inp, decoded)
        opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        self.CAD.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

        self.Flow = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(3,2), return_sequences=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation='relu')),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='relu')
        ])
        opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        self.Flow.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        print(self.Flow.summary())
        print(self.CAD.summary())

    def train_CAD(self, train, labels):
        self.CAD.fit(train, labels, epochs=5, batch_size=5)
        self.CAD.save('my_model.h5')

    def train_Flow(self, train, labels):
        self.Flow.fit(train, labels, epochs=5, batch_size=5)

    def evaluate(self, video):
        self.CAD = tf.keras.models.load_model('RMSProp_test.h5')
        video_path = os.path.join("/home/aswinvisva/watonomous/Jiqing Expressway Video", "IMG_" + video + ".MOV")
        print(video_path)

        cap = cv2.VideoCapture(video_path)

        data = []
        labels = []
        i = 1
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            label_path = os.path.join("/home/aswinvisva/watonomous", "Lane_Parameters", video, str(i) + ".txt")

            f = open(label_path, "r")

            label = np.zeros(frame.shape)
            matches = []
            for x in f:
                matches = matches + re.findall('\(.*?,.*?\)', x)

            for match in matches:
                tup = ast.literal_eval(match)
                if tup[0] >= 1920 or tup[1] >= 1080:
                    continue

                label[tup[1]][tup[0]] = 1

            i += 1

            if i == 5394:
                break

            label = cv2.resize(label, (896,896))
            frame = cv2.resize(frame, (896,896))
            frame = frame / 255
            frame = np.reshape(frame, (896,896, 1))
            label = np.reshape(label, (896,896, 1))
            predicted = self.CAD.predict(np.expand_dims(frame, axis=0))
            predicted = np.array(predicted) 
            predicted = predicted.reshape((896,896, 1))


            print(predicted)
            cv2.imshow('predicted', predicted)
            cv2.imshow('frame', frame)
            cv2.imshow('label', label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(frame.shape)
            print(label.shape)
#
# if __name__ == '__main__':
#     model = Model()


