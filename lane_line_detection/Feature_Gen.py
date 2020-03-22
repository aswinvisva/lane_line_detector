import numpy as np
import pandas as pd
import cv2
import os
import re
import ast
from lane_line_detection.Model import Model
from PIL import Image
from skimage.transform import resize

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
            matches =re.findall('\(.*?,.*?\)', x)

            matches = [ast.literal_eval(x) for x in matches]
            for idx,tup in enumerate(matches):
                print(idx, len(matches))
                if idx == len(matches) - 1:
                    break
                if tup[0] > 1918 or tup[1] > 1078:
                    continue

                label = cv2.line(label, tup, matches[idx+1], (255,255,255), 7)

        i+=1

        if i == 5394:
            break

        label = cv2.resize(label, (896,896), interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(frame, (896,896), interpolation=cv2.INTER_CUBIC)

        frame = frame/255
        frame = np.reshape(frame, (896,896,1))
        label = np.reshape(label, (896,896,1))

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

    data = data.reshape((5392, 896,896, 1))
    labels = labels.reshape((5392, 896,896, 1))


    return data, labels


if __name__ == '__main__':
    # videos = ['0254', '0261', '0263', '0255']
    #
    model = Model()
    model.model_fit_generator()
    # label_video('0254')
    #
    #
    model.CAD.save('backup.h5')
    #
    #
    # model.evaluate('0261')