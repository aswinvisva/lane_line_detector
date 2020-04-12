import ast
import codecs
import os
import random
import re
import time

import numpy as np

import cv2
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import ml_pipeline.models


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


class Generator:

    def evaluate_generator(self, **kwargs):

        option = kwargs.get("option", "driver_frame")
        image_size = kwargs.get("image_size", (256, 256))
        model = kwargs.get("model", "UNet")
        dont_show = kwargs.get("dont_show", True)
        shuffle = kwargs.get("shuffle", False)

        build_model = ml_pipeline.models.Models()

        loaded_model = tf.keras.models.load_model("Models/%s.h5" % model)

        if option == "driver_frame":
            return loaded_model.evaluate_generator(
                self.__label_video_generator_driver_frame(image_size, dont_show, shuffle))
        elif option == "jiqing":
            return loaded_model.evaluate_generator(
                self.__label_video_generator_jiqing_expressway(image_size, dont_show))
        elif option == "data_road":
            return loaded_model.evaluate_generator(self.__label_video_generator_data_road(image_size))
        else:
            print("Option does not exist!")

    def show_predictions(self, **kwargs):

        option = kwargs.get("option", "hardest_challenge")
        image_size = kwargs.get("image_size", (256, 256))
        model = kwargs.get("model", "UNet")

        build_model = ml_pipeline.models.Models()

        loaded_model = tf.keras.models.load_model("Models/%s.h5" % model)

        if option == "driver_frame":
            return self.__show_predictions_driver_frame(loaded_model, image_size)
        elif option == "jiqing":
            return self.__show_predictions_jiqing(loaded_model, image_size)
        elif option == "data_road":
            return self.__show_predictions_data_road(loaded_model, image_size)
        elif option == "hardest_challenge":
            return self.__show_predictions_hardest_challenge_vid(loaded_model, image_size)
        else:
            print("Option does not exist!")

    def data_generator(self, **kwargs):

        option = kwargs.get("option", "driver_frame")
        image_size = kwargs.get("image_size", (256, 256))
        train = kwargs.get("train", False)
        shuffle = kwargs.get("shuffle", True)

        if option == "driver_frame":
            return self.__label_video_generator_driver_frame(image_size, train, shuffle)
        elif option == "driver_frame_rolling_window":
            return self.__label_video_generator_driver_frame_rolling_window(image_size, train, shuffle)
        elif option == "jiqing":
            return self.__label_video_generator_jiqing_expressway(image_size, train)
        elif option == "data_road":
            return self.__label_video_generator_data_road(image_size)
        else:
            print("Option does not exist!")

    def __show_predictions_hardest_challenge_vid(self, CAD, image_size):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        video_path = "/home/aswinvisva/watonomous/lane_line_detection/harder_challenge_video.mp4"
        cap = cv2.VideoCapture(video_path)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if frame is None:
                break

            frame = cv2.resize(frame, image_size)
            original = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame = frame / 255
            frame = np.reshape(frame, (image_size[0], image_size[1], 1))

            predicted = CAD.predict(frame.reshape(1, image_size[0], image_size[1], 1))
            predicted = np.array(predicted)
            predicted = predicted.reshape((image_size[0], image_size[1], 1))
            predicted = np.array(predicted * 255, dtype='uint8')
            predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)
            white_lo = np.array([75, 75, 75])
            white_hi = np.array([255, 255, 255])
            mask = cv2.inRange(predicted, white_lo, white_hi)
            predicted[mask > 0] = (0, 0, 255)

            original = cv2.addWeighted(original, 0.5, predicted, 0.5, 0)

            cv2.imshow('predicted', predicted)
            cv2.imshow('frame', original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __show_predictions_data_road(self, model, image_size, rolling=False, window_size=3):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        window = []
        for root, dirs, files in os.walk('../data_road/testing/image_2'):
            for file in files:
                print(os.path.join('/home/aswinvisva/watonomous/data_road/testing/image_2', file))
                image = cv2.imread(os.path.join('/home/aswinvisva/watonomous/data_road/testing/image_2', file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.GaussianBlur(image, (3, 3), 0)
                # image = cv2.Canny(image, 75, 200)
                data_point = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
                data_point = data_point / 255
                original = data_point
                if not rolling:
                    data_point = np.reshape(data_point, (1, image_size[0], image_size[1], 1))
                    prediction = model.predict(data_point)
                    prediction = prediction.reshape((image_size[0], image_size[1], 1))
                    print(prediction)

                    cv2.imshow("Image", prediction)
                    cv2.imshow("Label", original)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    time.sleep(1)
                else:
                    if len(window) < window_size:
                        window.append(data_point)
                    else:
                        window.pop(0)
                        window.append(data_point)
                        window_np = np.array(window)
                        window_np = window_np.reshape(1, window_size, image_size[0], image_size[1], 1)
                        prediction = model.predict(window_np)
                        prediction = prediction.reshape((image_size[0], image_size[1], 1))

                        print(prediction)

                        cv2.imshow("Image", prediction)
                        cv2.imshow("Label", window[2])

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    def __show_predictions_jiqing(self, CAD, image_size, video="0261", window_size=3):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        video_path = os.path.join("/home/aswinvisva/watonomous/Jiqing Expressway Video", "IMG_" + video + ".MOV")

        cap = cv2.VideoCapture(video_path)

        while (True):

            ret, frame = cap.read()

            if frame is None:
                break

            frame = cv2.resize(frame, image_size)
            original = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame = frame / 255
            frame = np.reshape(frame, (image_size[0], image_size[1], 1))

            predicted = CAD.predict(frame.reshape(1, image_size[0], image_size[1], 1))
            predicted = np.array(predicted)
            predicted = predicted.reshape((image_size[0], image_size[1], 1))
            predicted = np.array(predicted * 255, dtype='uint8')
            predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)
            white_lo = np.array([75, 75, 75])
            white_hi = np.array([255, 255, 255])
            mask = cv2.inRange(predicted, white_lo, white_hi)
            predicted[mask > 0] = (0, 0, 255)

            original = cv2.addWeighted(original, 0.5, predicted, 0.5, 0)

            cv2.imshow('predicted', predicted)
            cv2.imshow('frame', original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def __show_predictions_driver_frame(self, model, image_size):
        i = 0

        for root, dirs, files in os.walk('../driver_frame'):
            # random.shuffle(files)
            random.shuffle(dirs)
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    f = open(path, "r")
                    file_name = re.search(".*?\.", file).group(0) + "jpg"
                    mat = cv2.imread(os.path.join(root, file_name))
                    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                    label = np.zeros((mat.shape[0], mat.shape[1], 1))

                    line_no = 0
                    for line in f:
                        coords = line.split(" ")
                        index = 0
                        line_no = line_no + 1

                        while (index <= (len(coords) - 4)):
                            pointA = (round(float(coords[index])), round(float(coords[index + 1])))
                            pointB = (round(float(coords[index + 2])), round(float(coords[index + 3])))
                            label = cv2.line(label, pointA, pointB, (255, 255, 255), 25)

                            index = index + 2

                    i = i + 1

                    # mat = mat[250:, :, :]
                    mat = cv2.GaussianBlur(mat, (3, 3), 0)
                    mat = cv2.resize(mat, image_size)
                    label = cv2.resize(label, image_size)

                    data_point = np.reshape(mat, (1, image_size[0], image_size[1], 1))
                    prediction = model.predict(data_point)
                    prediction = prediction.reshape((image_size[0], image_size[1], 1))
                    prediction = prediction * 255

                    cv2.imshow("Prediction", prediction)
                    cv2.imshow("Image", mat)
                    cv2.imshow("Label", label)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    def __label_video_generator_data_road(self, image_size):
        data = []
        for root, dirs, files in os.walk('../data_road/training/image_2'):
            for file in files:
                print(os.path.join('/home/aswinvisva/watonomous/data_road/training/image_2', file))
                image = cv2.imread(os.path.join('/home/aswinvisva/watonomous/data_road/training/image_2', file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
                image = image / 255
                image = np.reshape(image, (image_size[0], image_size[1], 1))
                data.append(image)

        labels = []
        for root, dirs, files in os.walk('../data_road/training/gt_image_2'):
            for file in files:
                print(os.path.join('/home/aswinvisva/watonomous/data_road/training/gt_image_2', file))
                image = cv2.imread(os.path.join('/home/aswinvisva/watonomous/data_road/training/gt_image_2', file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (image_size[0], image_size[1]), interpolation=cv2.INTER_CUBIC)
                image = image / 255
                image = np.reshape(image, (image_size[0], image_size[1], 1))
                labels.append(image)

        return np.array(data), np.array(labels)

    def __label_video_generator_jiqing_expressway(self, image_size, train, videos=['0254', '0255', '0261', '0263']):

        for video in videos:
            video_path = os.path.join("/home/aswinvisva/watonomous/Jiqing Expressway Video", "IMG_" + video + ".MOV")
            print(video_path)

            cap = cv2.VideoCapture(video_path)

            i = 1
            while (True):
                # Capture frame-by-frame
                ret, frame = cap.read()

                label_path = os.path.join("/home/aswinvisva/watonomous", "Lane_Parameters", video, str(i) + ".txt")

                f = open(label_path, "r")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                label = np.zeros((frame.shape[0], frame.shape[1], 1))

                for x in f:
                    matches = re.findall('\(.*?,.*?\)', x)

                    matches = [ast.literal_eval(x) for x in matches]
                    for idx, tup in enumerate(matches):
                        if idx == len(matches) - 1:
                            break
                        if tup[0] > 1918 or tup[1] > 1078:
                            continue

                        label = cv2.line(label, tup, matches[idx + 1], (255, 255, 255), 25)

                i += 1

                if i == 5394:
                    break

                label = cv2.resize(label, image_size, interpolation=cv2.INTER_CUBIC)
                frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_CUBIC)

                label = np.reshape(label, (image_size[0], image_size[1], 1))
                frame = np.reshape(frame, (image_size[0], image_size[1], 1))
                frame = frame / 255
                label = label / 255

                if not train:
                    cv2.imshow('frame', frame)
                    cv2.imshow('label', label)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                yield (np.array(frame).reshape(1, image_size[0], image_size[1], 1),
                       np.array(label).reshape(1, image_size[0], image_size[1], 1))

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

    def __label_video_generator_driver_frame_rolling_window(self,
                                                            image_size,
                                                            train,
                                                            shuffle,
                                                            window_size=3,
                                                            batch_size=1):

        data = []
        labels = []

        for root, dirs, files in os.walk('../driver_frame'):
            dir_root = root
            files = sorted(files)
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_name = re.search(".*?\.", file).group(0) + "jpg"
                    data_path = os.path.join(root, file_name)
                    data.append(data_path)
                    labels.append(file_path)

        batch_data = []
        batch_label = []
        windows = []
        window_labels = []

        print(data[0:20])
        print(labels[0:20])

        for i in range(len(data) - window_size - 1):
            window = []
            for x in range(window_size):
                window.append(data[i+x])

            windows.append(window)
            window_labels.append(labels[i + window_size])

        print(windows[0:20])
        print(window_labels[0:20])
        shuffle_in_unison(window_labels, windows)
        print(windows[0:20])
        print(window_labels[0:20])
        for i in range(len(windows)):

            window = []
            label = []

            for x in windows[i]:
                file_name = x
                mat = cv2.imread(file_name)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                label = np.zeros((mat.shape[0], mat.shape[1], 1))
                mat = cv2.GaussianBlur(mat, (3, 3), 0)
                mat = cv2.resize(mat, image_size)
                mat = mat.reshape((image_size[0], image_size[1], 1))
                mat = mat / 255
                window.append(mat)

            path = window_labels[i]

            f = open(path, "r")

            line_no = 0
            for line in f:
                line = str(line)
                coords = line.split(" ")
                index = 0
                line_no = line_no + 1

                while (index <= (len(coords) - 4)):
                    pointA = (round(float(coords[index])), round(float(coords[index + 1])))
                    pointB = (round(float(coords[index + 2])), round(float(coords[index + 3])))
                    label = cv2.line(label, pointA, pointB, (255, 255, 255), 25)

                    index = index + 2

            label = cv2.resize(label, image_size)
            label = label.reshape((image_size[0], image_size[1], 1))
            label = label / 255

            batch_data.append(window)
            batch_label.append(label)

            batch_data_np = np.array(batch_data)
            batch_label_np = np.array(batch_label)

            if len(batch_data) == batch_size:
                yield (batch_data_np.reshape(batch_size, window_size, image_size[0], image_size[1], 1),
                       batch_label_np.reshape(batch_size, image_size[0], image_size[1], 1))
                batch_data = []
                batch_label = []

    def __label_video_generator_driver_frame(self, image_size, train, shuffle, batch_size=5):
        i = 0
        batch_data = []
        batch_label = []

        for root, dirs, files in os.walk('../driver_frame'):
            if shuffle:
                random.shuffle(files)
                random.shuffle(dirs)

            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    f = open(path, "r")
                    file_name = re.search(".*?\.", file).group(0) + "jpg"
                    mat = cv2.imread(os.path.join(root, file_name))
                    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                    label = np.zeros((mat.shape[0], mat.shape[1], 1))

                    line_no = 0
                    for line in f:
                        coords = line.split(" ")
                        index = 0
                        line_no = line_no + 1

                        while (index <= (len(coords) - 4)):
                            pointA = (round(float(coords[index])), round(float(coords[index + 1])))
                            pointB = (round(float(coords[index + 2])), round(float(coords[index + 3])))
                            label = cv2.line(label, pointA, pointB, (255, 255, 255), 25)

                            index = index + 2

                    i = i + 1

                    mat = cv2.GaussianBlur(mat, (3, 3), 0)
                    mat = cv2.resize(mat, image_size)
                    label = cv2.resize(label, image_size)
                    angle = random.randint(-20, 20)

                    mat = rotate_image(mat, angle)
                    label = rotate_image(label, angle)

                    if not train:
                        cv2.imshow("Image", mat)
                        cv2.imshow("Label", label)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    mat = mat.reshape((image_size[0], image_size[1], 1))
                    label = label.reshape((image_size[0], image_size[1], 1))
                    mat = mat / 255
                    label = label / 255

                    batch_data.append(mat)
                    batch_label.append(label)

                    batch_data_np = np.array(batch_data)
                    batch_label_np = np.array(batch_label)

                    shuffle_in_unison(batch_data_np, batch_label_np)

                    if len(batch_data) == batch_size:
                        yield (batch_data_np.reshape(batch_size, image_size[0], image_size[1], 1),
                               batch_label_np.reshape(batch_size, image_size[0], image_size[1], 1))
                        batch_data = []
                        batch_label = []
