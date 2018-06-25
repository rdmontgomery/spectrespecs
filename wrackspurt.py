import cv2
import io
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import threading


def instantiate_camera(resolution=(640, 480), framerate=24):
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = framerate
    return camera


def get_frame(camera, stream):
    camera.capture(stream, format='jpeg')
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, 1)
    cv2.flip(frame, 1, frame)

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(grayscale)

    grayscale = cv2.GaussianBlur(grayscale, (9, 9), 1.5)

    dilation_params = (5, 5)
    dilate_kernel = np.ones(dilation_params, np.uint8)
    grayscale = cv2.dilate(grayscale, dilate_kernel, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grayscale = clahe.apply(grayscale)

    return grayscale


def findWand(grayscale):
    output = grayscale.copy()

    coords = cv2.HoughCircles(grayscale,
                              cv2.HOUGH_GRADIENT,
                              dp=3,
                              minDist=50,
                              param1=240,
                              param2=8,
                              minRadius=4,
                              maxRadius=15)
    
    circles = coords
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (128, 0, 0), 2)
            cv2.rectangle(output, (x-1, y-1), (x+1, y+1), (0, 0, 0), -1)

    # threading.Timer(1, findWand, [grayscale]).start()
    return output, coords


def main():
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10,
                               0.03))

    camera = instantiate_camera()
    prev_frame = None
    with io.BytesIO() as stream:
        while cv2.waitKey(1) & 0xFF != ord('q'):
            next_frame = get_frame(camera, stream)
            output, old_points = findWand(next_frame)
            if old_points is not None:
                old_points.shape = (old_points.shape[1], 1, old_points.shape[2])
                #print(old_points)
                old_points = old_points[:, :, 0:2][:, 0]
                #print(old_points)
                #print(old_points[:, 0])

            if prev_frame is not None:
                pass
                # print(prev_frame)
                new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame,
                                                                   next_frame,
                                                                   old_points,
                                                                   None,
                                                                    **lk_params)
                print(new_points, status, err)
            prev_frame = next_frame

            cv2.imshow('frame', np.hstack([prev_frame, output]))
            stream.truncate()
            stream.seek(0)

        cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()

