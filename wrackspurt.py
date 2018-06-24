import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import threading


def instantiate_camera(resolution=(640, 480), framerate=32):
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = framerate
    return camera


def get_frame(camera, stream):
    camera.capture(stream, 'bgr', use_video_port=True)
    # stream.array should contain image data in BGR order
    grayscale = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

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

    circles = cv2.HoughCircles(grayscale,
                               cv2.HOUGH_GRADIENT,
                               dp=3,
                               minDist=50,
                               param1=240,
                               param2=8,
                               minRadius=4,
                               maxRadius=15)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (128, 0, 0), 2)
            cv2.rectangle(output, (x-1, y-1), (x+1, y+1), (0, 0, 0), -1)

    # threading.Timer(1, findWand, [grayscale]).start()
    return output


def main():
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10,
                               0.03))

    camera = instantiate_camera()
    with PiRGBArray(camera) as stream:
        while cv2.waitKey(1) & 0xFF != ord('q'):
            prev_frame = get_frame(camera, stream)
            output = findWand(prev_frame)

            cv2.imshow('frame', np.hstack([prev_frame, output]))
            stream.truncate(0)

        cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()

