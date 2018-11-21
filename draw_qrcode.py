import cv2
import pyqrcode
import create_background
from PIL import Image
import GetVectors
import triad2D
import numpy as np
import calibrate_camera
from sympy.matrices import *
import pyboof as pb

def draw_3D_axis(im, img_origin_pts, img_axis_pts):
    """ use draw function to draw 3d axis on 2d images given 2D and 3D point correspondence """
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[0].ravel()), (255, 0, 0), 5)
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[1].ravel()), (0, 255, 0), 5)
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[2].ravel()), (0, 0, 255), 5)
    return im


#               make instance of QRCode Class
qr_object = pyqrcode.create('skripsie 2018', error='H', version=2)

#               Make output image
qr_img = "qr_img.png"
qr_object.png(qr_img, scale=2)
pill_image = Image.open(qr_img)
background = create_background.create_background(600, 600)


#               Get rotation image
basewidth = 200
wpercent = (basewidth/float(pill_image.size[0]))
hsize = int(float(pill_image.size[1]) * float(wpercent))
rotated_qr_image = pill_image.resize((basewidth, hsize), Image.ANTIALIAS).rotate(30, expand=True)
rotated_qr_image2 = pill_image.resize((basewidth, hsize), Image.ANTIALIAS).rotate(60, expand=True)


#               Add qrcodes to background
background.paste(rotated_qr_image, (0, 0))
background.paste(rotated_qr_image2, (300, 300))

#               Save and reopen with opencv
background.save("background_qrcodes.png")

#               Detect QR-codes in images
#               new method
fname = "background_qrcodes.png"
detector = pb.FactoryFiducial(np.uint8).qrcode()
image = pb.load_single_band(fname, np.uint8)
detector.detect(image)
print("Detected a total of {} QR Codes".format(len(detector.detections)))
for qr in detector.detections:
    print("Message: "+qr.message)
    print("     at: "+str(qr.bounds))


#               Send QR-corner points to GetCoords function
object_points_1 = detector.detections[0].bounds
object_points_2 = detector.detections[1].bounds

Q1 = np.array([
    [object_points_1.vertexes[3].x, object_points_1.vertexes[3].y, 0],
    [object_points_1.vertexes[2].x, object_points_1.vertexes[2].y, 0],
    [object_points_1.vertexes[0].x, object_points_1.vertexes[0].y, 0],
    [object_points_1.vertexes[3].x, object_points_1.vertexes[3].y, 1]
], dtype=np.float64)

Q2 = np.array([
    [object_points_2.vertexes[3].x, object_points_2.vertexes[3].y, 0],
    [object_points_2.vertexes[2].x, object_points_2.vertexes[2].y, 0],
    [object_points_2.vertexes[0].x, object_points_2.vertexes[0].y, 0],
    [object_points_2.vertexes[3].x, object_points_2.vertexes[3].y, 1]
], dtype=np.float64)

#               Call Get Vectors function
A, B = GetVectors.get_coords(Q1, Q2)
print("A:", A)
print("B", B)

AdcmB, trueAdcmB, BdcmA, trueBdcmA = triad2D.triad2d(A, B)

print("Trying solving with Perspective n problem")
# solve problem with Perspective n problem

# get corresponding image and object points of A and B respectively
A_objpts = np.array([
    [0, 0, 0],
    [20, 0, 0],
    [0, 20, 0],
    [20, 20, 0]
], dtype=np.float64)

A_imgpts = np.array([
    [object_points_1.vertexes[3].x, object_points_1.vertexes[3].y],
    [object_points_1.vertexes[2].x, object_points_1.vertexes[2].y],
    [object_points_1.vertexes[0].x, object_points_1.vertexes[0].y],
    [object_points_1.vertexes[1].x, object_points_1.vertexes[1].y]
], dtype=np.float64)

B_objpts = np.array([
    [0, 0, 0],
    [20, 0, 0],
    [0, 20, 0],
    [20, 20, 0]
], dtype=np.float64)

B_imgpts = np.array([
    [object_points_2.vertexes[3].x, object_points_2.vertexes[3].y],
    [object_points_2.vertexes[2].x, object_points_2.vertexes[2].y],
    [object_points_2.vertexes[0].x, object_points_2.vertexes[0].y],
    [object_points_2.vertexes[1].x, object_points_2.vertexes[1].y]
], dtype=np.float64)

# load camera instrinsic
mtx, dist = calibrate_camera.get_calibration_results()
print("mtx", mtx)
print("dist", mtx)

# use solvePnP on both A and B to get rotation with respect to camera coordinate system
retval, A_rvecs, A_tvecs = cv2.solvePnP(A_objpts, A_imgpts, mtx, dist)
retval, B_rvecs, B_tvecs = cv2.solvePnP(B_objpts, B_imgpts, mtx, dist)

#               draw axis on qrcodes
#               Initialize axis 3D object points
axis = np.float32([[20, 0, 0],  # x-axis
                   [0, 20, 0],  # y-axis
                   [0, 0, 20]  # z-axis
                   ]).reshape(-1, 3)

#               Load image with opencv
img = cv2.imread(fname)
A_axis_imgpts, jac = cv2.projectPoints(axis, A_rvecs, A_tvecs, mtx, dist)
draw_3D_axis(img, (object_points_1.vertexes[3].x, object_points_1.vertexes[3].y), A_axis_imgpts)

B_axis_imgpts, jac = cv2.projectPoints(axis, B_rvecs, B_tvecs, mtx, dist)
draw_3D_axis(img, (object_points_2.vertexes[3].x, object_points_2.vertexes[3].y), B_axis_imgpts)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#   get rotation angle
A_angle_rad = np.sqrt((A_rvecs[0]*A_rvecs[0]) + (A_rvecs[1]*A_rvecs[1]) + (A_rvecs[2]*A_rvecs[2]))
B_angle_rad = np.sqrt((B_rvecs[0]*B_rvecs[0]) + (B_rvecs[1]*B_rvecs[1]) + (B_rvecs[2]*B_rvecs[2]))

#   use Rodrigues to transform pvr to dcm
AdcmO,_ = cv2.Rodrigues(src=A_rvecs)
BdcmO,_ = cv2.Rodrigues(src=B_rvecs)

#   cast to matrix to perform dot product
AdcmO = Matrix(AdcmO)
BdcmO = Matrix(BdcmO)

AdcmB1 = BdcmO*AdcmO.transpose()
BdcmA1 = AdcmO*BdcmO.transpose()

#               Get Error Matrix
EdcmAB = AdcmB1 * trueAdcmB.transpose()
EdcmBA = BdcmA1 * trueBdcmA.transpose()

print("AdcmB1:", AdcmB1)
print("BdcmA1:", BdcmA1)
print("EdcmAB:",EdcmAB)
print("EdcmBA:",EdcmBA)








