import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np

import ml_pipeline.feature_generator


class Models:

    def __init__(self, **kwargs):
        model = kwargs.get("model", "unet")

        self.model_name = model
        self.model = None

        if model == "unet":
            self.model = self.build_model_unet()
        elif model == "convlstm":
            self.model = self.build_model_convlstm()
        elif model == "BCDU_net_D3":
            self.build_model_unet()
            self.model = self.build_model_BCDU_net_D3()
        else:
            print("Option does not exist!")

    def build_model_BCDU_net_D3(self, input_size=(256, 256, 1)):
        N = input_size[0]
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # D1
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4_1 = Dropout(0.5)(conv4_1)
        # D2
        conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4_1)
        conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_2)
        conv4_2 = Dropout(0.5)(conv4_2)
        # D3
        merge_dense = concatenate([conv4_2, drop4_1], axis=3)
        conv4_3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_dense)
        conv4_3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_3)
        drop4_3 = Dropout(0.5)(conv4_3)

        up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
        up6 = BatchNormalization(axis=3)(up6)
        up6 = Activation('relu')(up6)

        x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
        x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
        merge6 = concatenate([x1, x2], axis=1)
        merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                            kernel_initializer='he_normal')(merge6)

        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
        up7 = BatchNormalization(axis=3)(up7)
        up7 = Activation('relu')(up7)

        x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
        x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
        merge7 = concatenate([x1, x2], axis=1)
        merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                            kernel_initializer='he_normal')(merge7)

        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
        up8 = BatchNormalization(axis=3)(up8)
        up8 = Activation('relu')(up8)

        x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
        x2 = Reshape(target_shape=(1, N, N, 64))(up8)
        merge8 = concatenate([x1, x2], axis=1)
        merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                            kernel_initializer='he_normal')(merge8)

        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

        model = Model(inputs, conv9)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_model_convlstm(self, time_steps=3, image_size=(256,256)):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        inputs = Input((time_steps, image_size[0], image_size[1], 1))
        convlstm1 = ConvLSTM2D(filters=32, activation="relu", kernel_size=(3, 3),
                               padding='same', return_sequences=True)(inputs)
        convlstm1 = ConvLSTM2D(filters=32, activation="relu", kernel_size=(3, 3),
                               padding='same', return_sequences=True)(convlstm1)
        batchnormalization2 = BatchNormalization()(convlstm1)

        convlstm3 = ConvLSTM2D(filters=64, activation="relu", kernel_size=(3, 3),
                               padding='same', return_sequences=True)(batchnormalization2)
        convlstm3 = ConvLSTM2D(filters=64, activation="relu", kernel_size=(3, 3),
                               padding='same', return_sequences=False)(convlstm3)
        batchnormalization3 = BatchNormalization()(convlstm3)

        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batchnormalization3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

        maxpooling5 = MaxPooling2D(pool_size=(2, 2))(conv4)
        batchnormalization5 = BatchNormalization()(maxpooling5)

        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batchnormalization5)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        maxpooling7 = MaxPooling2D(pool_size=(2, 2))(conv6)
        batchnormalization7 = BatchNormalization()(maxpooling7)

        conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(batchnormalization7)
        conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        drop9 = Dropout(0.5)(conv8)

        maxpooling10 = MaxPooling2D(pool_size=(2, 2))(drop9)
        batchnormalization10 = BatchNormalization()(maxpooling10)
        drop10 = Dropout(0.5)(batchnormalization10)

        up10 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop10))

        merge6 = concatenate([drop9, up10], axis=3)
        conv9 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv9 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        up10 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv9))
        merge10 = concatenate([conv6, up10], axis=3)
        conv10 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
        conv10 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

        up11 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv10))
        merge11 = concatenate([conv4, up11], axis=3)
        conv11 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
        conv11 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

        merge12 = concatenate([convlstm3, conv11], axis=3)
        conv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
        conv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
        conv12 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
        conv13 = Conv2D(1, 1, activation='sigmoid')(conv12)

        model = Model(inputs, conv13)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

        model.summary()

        return model

    def build_model_unet(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        inputs = Input((256, 256, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

        return model

    def fit_model(self, shuffle=True):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0.5,
                                  write_graph=True, write_images=True)

        gen = ml_pipeline.feature_generator.Generator()

        self.model.fit(x=gen.data_generator(train=True, shuffle=shuffle), epochs=250, callbacks=[tensorboard],
                       steps_per_epoch=150,
                       shuffle=True)

        # tf.keras.models.save_model(self.model, 'Models/convlstm.h5')
        tf.keras.models.save_model(self.model, 'Models/%s.h5' % self.model_name)
