import os
import random
import re

import cv2
import numpy as np

class ImageGenerator:

    def __init__(self, train=True):

        d, l = self.label_video_generator_driver_frame(train=train)

        self.data = d
        self.labels = l
        self.batch_size = 13

        print(self.data)

        self.shuffle_in_unison(self.data, self.labels)

        print(self.data)

    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def generate(self):
        batch_data = []
        batch_label = []
        for x, y in np.nditer([self.data, self.labels]):
            batch_data.append(x)
            batch_label.append(y)

            if len(batch_data) == self.batch_size:
                yield (np.array(batch_data).reshape(self.batch_size, 240, 426, 3),
                       np.array(batch_label).reshape(self.batch_size, 30, 30, 1))
                batch_data = []
                batch_label = []


    def label_video_generator_driver_frame(self, train=True):
        i = 0
        batch_data = []
        batch_label = []

        for root, dirs, files in os.walk('../driver_frame'):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    f = open(path, "r")
                    file_name = re.search(".*?\.", file).group(0) + "jpg"
                    mat = cv2.imread(os.path.join(root, file_name))
                    # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                    label = np.zeros((mat.shape[0], mat.shape[1], 1))
                    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                    # print(mat.shape)
                    for line in f:
                        coords = line.split(" ")
                        index = 0
                        while (index <= (len(coords) - 4)):
                            pointA = (round(float(coords[index])), round(float(coords[index + 1])))
                            pointB = (round(float(coords[index + 2])), round(float(coords[index + 3])))
                            label = cv2.line(label, pointA, pointB, (255, 255, 255), 25)

                            index = index + 2

                    i = i + 1

                    # print(len(batch_label))\
                    mat = mat[375:, :, :]
                    mat = cv2.GaussianBlur(mat, (3, 3), 0)
                    # mat = cv2.Canny(mat, 75, 200)
                    mat = cv2.resize(mat, (426, 240))
                    label = cv2.resize(label, (30, 30))
                    angle = random.randint(-5, 5)

                    mat = self.rotate_image(mat, angle)
                    label = self.rotate_image(label, angle)

                    if not train:
                        cv2.imshow("Image", mat)
                        cv2.imshow("Label", label)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    mat = mat.reshape((240, 426, 3))
                    label = label.reshape((30, 30, 1))
                    mat = mat / 255
                    label = label / 255

                    batch_data.append(mat)
                    batch_label.append(label)

        return np.array(batch_data), np.array(batch_label)