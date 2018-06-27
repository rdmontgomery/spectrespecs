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
    camera.capture(stream, 'bgr', use_video_port=True)
    grayscale = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

    # Process image to make points of light more distinct as circles
    cv2.equalizeHist(grayscale)
    grayscale = cv2.GaussianBlur(grayscale, (9, 9), 1.5)
    dilation_params = (5, 5)
    dilate_kernel = np.ones(dilation_params, np.uint8)
    grayscale = cv2.dilate(grayscale, dilate_kernel, iterations=1)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grayscale = clahe.apply(grayscale)

    return grayscale


def findWand(grayscale):
    coords = cv2.HoughCircles(grayscale,
                              cv2.HOUGH_GRADIENT,
                              dp=3,
                              minDist=50,
                              param1=240,
                              param2=8,
                              minRadius=4,
                              maxRadius=15)

    if coords is not None:
        # Draw center and circle around found points of light
        circles = np.round(coords[0, :]).astype('int')
        for (x, y, r) in circles:
            cv2.circle(grayscale, (x, y), r, (128, 0, 0), 2)
            cv2.rectangle(grayscale, (x-1, y-1), (x+1, y+1), (0, 0, 0), -1)

        # The following are necessary for calcOpticalFlowPyrLK
        # Coordinates need shape (n, 1, 2), where n is the number of points
        coords.shape = (coords.shape[1], 1, coords.shape[2])
        # Select just the (x, y) coords, removing the radius
        coords = coords[:, :, :2]
        coords = coords.astype(np.float32)

    return coords


def trackWand(prev_frame, next_frame, old_points):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10,
                               0.03))
    mask = np.zeros_like(prev_frame)

    new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame,
                                                       next_frame,
                                                       old_points,
                                                       None,
                                                        **lk_params)
    print(new_points, status, err)
    # Select points with a good status
    valid_old = old_points[status==1]
    valid_new = new_points[status==1]
    if len(valid_new) > 0:
        for i, (new, old) in enumerate(zip(valid_new, valid_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(next_frame, (a, b), (c, d), (128,0,0), 2)
            cv2.circle(next_frame, (a, b), 5, (255,0,0), -1)


def main():
    camera = instantiate_camera()
    prev_frame = None
    with PiRGBArray(camera) as stream:
        while cv2.waitKey(1) & 0xFF != ord('q'):
            next_frame = get_frame(camera, stream)
            old_points = findWand(next_frame)

            if prev_frame is not None and old_points is not None:
                trackWand(prev_frame, next_frame, old_points)
            prev_frame = next_frame

            cv2.imshow('frame', np.hstack([next_frame]))
            stream.truncate()
            stream.seek(0)

        cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()

