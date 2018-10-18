from checkerboards import CheckerBoard
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

x1 = CheckerBoard()
x2 = CheckerBoard()

x1.draw_chessboard()
x2.draw_chessboard()

roi_x1 = x1.chessboard
roi_x2 = x2.chessboard
print(roi_x2.shape[0])


background = np.zeros((1000, 1000), np.uint8)

x1_str = (int(np.random.uniform(0, 800)), int(np.random.uniform(0, 800)))
x2_str = (int(np.random.uniform(0, 800)), int(np.random.uniform(0, 800)))
print(x1_str, x2_str)

background[x1_str[0]:x1_str[0]+roi_x1.shape[0], x1_str[1]:x1_str[1]+roi_x1.shape[1]] = roi_x1
background[x2_str[0]:x2_str[0]+roi_x2.shape[0], x2_str[1]:x2_str[1]+roi_x2.shape[1]] = roi_x2


# nx = 7
# ny = 7
# ret, corners = cv2.findChessboardCorners(background, (nx, ny), None)
# cv2.drawChessboardCorners(background, (nx, ny), corners, ret)

img = cv2.imread("/test/download.jpeg", cv2.IMREAD_GRAYSCALE)
if img is None:  # Check for invalid input
    print("Could not open or find the image")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nx = 14
ny = 14
ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

# Draw and display the corners

cv2.imshow("new", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

