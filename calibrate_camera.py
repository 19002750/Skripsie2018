import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0)......(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world
imgpoints = []  # 2d points in image plane

img_mask = 'data/left*.jpg'  # default
img_names = glob.glob(img_mask)
shape = []

for fname in img_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # if found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        shape = gray.shape[::-1]

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, 1), criteria)
        imgpoints.append(corners2)

        # draw and display the corneres
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# now calibrate the camera
ret, mtx, dist, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

# draw function for pose estimation
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

img_mask = 'data/left*.jpg'  # default
img_names = glob.glob(img_mask)

for fname in img_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

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