import cv2
import numpy as np
from operator import itemgetter
import itertools
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class Lanes:

    def __init__(self, path):
        self.path = path

    def canny_edge_detector(self, image):

        # Convert the image color to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Reduce noise from the image
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                try:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                except:
                    pass

        return line_image

    def region_of_interest(self, image):
        height = image.shape[0]
        polygons = np.array([
            [(200, height), (1100, height), (550, 250)]
        ])
        mask = np.zeros_like(image)

        # Fill poly-function deals with multiple polygon
        cv2.fillPoly(mask, polygons, 255)

        # Bitwise operation between canny image and mask image
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def create_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            # It will fit the polynomial and the intercept and slope
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.create_coordinates(image, left_fit_average)
        right_line = self.create_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def process(self):
        cap = cv2.VideoCapture(self.path)
        np.random.seed(0)
        color = {}
        for i in range(7 + 1):
            color[i] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        while (True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            canny_image = self.canny_edge_detector(frame)
            cropped_image = self.region_of_interest(canny_image)

            lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                                    np.array([]), minLineLength=40,
                                    maxLineGap=5)

            linez = [line.reshape(4) for line in lines]
            # linez = list(itertools.chain(*linez))

            # linez = sorted(linez, key=lambda x: x[1])
            X = StandardScaler().fit_transform(linez)
            print(linez)

            db = DBSCAN(eps=0.01, min_samples=5, metric="correlation")

            clusters = db.fit_predict(X)

            end = []
            for indx, x in enumerate(linez):
                if clusters[indx] != -1:
                    end.append([x,clusters[indx]])

            # averaged_lines = self.average_slope_intercept(frame, lines)
            # line_image = self.display_lines(frame, averaged_lines)
            # combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            # cv2.imshow("results", combo_image)

            for line in end:
                # try:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]
                id = line[1]


                cv2.line(frame, (x1, y1), (x2, y2), color[id], 10)

                    # frame = cv2.circle(frame, (x, y), 5, (color, 0, 255), -1)
                # except:
                #     print("bl")
                #     pass


            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    lanes = Lanes('harder_challenge_video.mp4')
    lanes.process()