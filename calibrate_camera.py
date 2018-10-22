import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import glob
import time

# Initial varaibles #
num_boards = 20
pattern_size = (7, 9)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0)......(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world
imgpoints = []  # 2d points in image plane


# Functions #
def take_calibrate_pictures():
    """take photos of chessboards to calibrate camera"""

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.vflip = True
    rawCapture = PiRGBArray(camera)

    # allow camera to warmup
    time.sleep(0.1)

    # grab an image of checkerboard from the camera for num_boards time
    for imagenum in range(1, (num_boards + 1)):
        time.sleep(0.1)
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        path = '/home/pi/PycharmProjects/Skripsie2018/data/'
        cv2.imwrite(str(path) + "image" + str(imagenum) + ".png", image)
        cv2.imshow("capture_image", image)
        cv2.waitKey(0)
        rawCapture.truncate(0)
        cv2.destroyAllWindows()


# draw function for pose estimation
def draw(im, chessboards_corners, img_pts):
    corner = tuple(chessboards_corners[0].ravel())
    im = cv2.line(im, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    im = cv2.line(im, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    im = cv2.line(im, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    return im


# Main loop
# calibrate with already existing images
img_mask = 'data/image*.png'
img_names = glob.glob(img_mask)

if len(img_names) == 0:
    print("No precaptured images found, taking pictures now")
    take_calibrate_pictures()
    img_mask = 'data/image*.jpg'
    img_names = glob.glob(img_mask)

print("Picture ready and calibration will start")
imagesize = []

# calibrate pi camera with taken pictures
for fname in img_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        imagesize.append(gray.shape[::-1])

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, 1), criteria)
        imgpoints.append(corners2)

        # draw and display the corneres
        img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)


# now calibrate the camera
ret, mtx, dist, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, imagesize[1], None, None)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

for fname in img_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # if found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, 1), criteria)

        # find the rotation and translation vectors
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        print("rvecs:", rvecs)
        mag = np.sqrt((rvecs[0]*rvecs[0])+(rvecs[1]*rvecs[1])+(rvecs[2]*rvecs[2]))
        print("angle:", mag)
        print("tvecs:", tvecs)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        cv2.imshow("img", img)
        cv2.waitKey(0)


cv2.destroyAllWindows()
