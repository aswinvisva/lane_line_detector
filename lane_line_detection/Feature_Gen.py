import numpy as np
import pandas as pd
import cv2
import os
import re
import ast
from lane_line_detection.Model import Model

def label_video(video):
    video_path = os.path.join("/home/aswinvisva/watonomous/Jiqing Expressway Video", "IMG_"+video+".MOV")
    print(video_path)

    cap = cv2.VideoCapture(video_path)
    data = []
    labels = []
    i = 1
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        label_path = os.path.join("/home/aswinvisva/watonomous", "Lane_Parameters", video, str(i)+".txt")

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

        i+=1


        if i == 5394:
            break

        label = cv2.resize(label, (112,112))
        frame = cv2.resize(frame, (112,112))
        frame = frame/255
        frame = np.reshape(frame, (112,112,1))
        label = np.reshape(label, (112,112,1))

        cv2.imshow('frame', frame)
        cv2.imshow('label', label)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        print(frame.shape)
        print(label.shape)

        data.append(frame)
        labels.append(label)

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    data = np.array(data)
    labels = np.array(labels)

    data = data.reshape((5392, 112, 112, 1))
    labels = labels.reshape((5392, 112,112, 1))


    return data, labels

def label_video_generator():

    videos = ['0254', '0261', '0263', '0255']

    for video in videos:
        video_path = os.path.join("/home/aswinvisva/watonomous/Jiqing Expressway Video", "IMG_"+video+".MOV")
        print(video_path)

        cap = cv2.VideoCapture(video_path)
        frame = []
        label = []
        i = 1
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            label_path = os.path.join("/home/aswinvisva/watonomous", "Lane_Parameters", video, str(i)+".txt")

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

            i+=1


            if i == 5394:
                break

            label = cv2.resize(label, (112,112))
            frame = cv2.resize(frame, (112,112))
            frame = frame/255
            frame = np.reshape(frame, (1, 112,112,1))
            label = np.reshape(label, (1, 112,112,1))
            print(frame.shape)

            yield (frame, label)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    videos = ['0254', '0261', '0263', '0255']

    model = Model()
    # model.CAD.fit_generator(label_video_generator(), steps_per_epoch=4313, epochs=4)
    #
    #
    # model.CAD.save('my_model.h5')


    model.evaluate('0255')