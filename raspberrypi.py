import pyboof as pb
import calibrate_camera
import numpy as np
import cv2
import pyqrcode
from PIL import Image
from sympy.matrices import *

#               Function Prototype
def draw_3D_axis(im, img_origin_pts, img_axis_pts):
    """ use draw function to draw 3d axis on 2d images given 2D and 3D point correspondence """
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[0].ravel()), (255, 0, 0), 5)
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[1].ravel()), (0, 255, 0), 5)
    im = cv2.line(im, (int(img_origin_pts[0]), int(img_origin_pts[1])), tuple(img_axis_pts[2].ravel()), (0, 0, 255), 5)
    return im



#               Start Detection

pb.init_memmap()

#               Initialize object points
A_objpts = np.array([
    [0, 0, 0],  # origin
    [20, 0, 0],  # x-axis
    [0, 20, 0],  # y-axis
    [20, 20, 0]  # (1,1,0) point
], dtype=np.float64)

B_objpts = np.array([
    [0, 0, 0],  # origin
    [20, 0, 0],  # x-axis
    [0, 20, 0],  # y-axis
    [20, 20, 0]  # (1,1,0) point
], dtype=np.float64)

#               Get Camera matrix
mtx, dist = calibrate_camera.get_calibration_results()

#               Initialize axis 3D object points
axis = np.float32([[20, 0, 0],  # x-axis
                   [0, 20, 0],  # y-axis
                   [0, 0, 20]  # z-axis
                   ]).reshape(-1, 3)

#               Get image
#fname = 'background_qrcodes.png'
fname = 'data2/fake8.png'
detector = pb.FactoryFiducial(np.uint8).qrcode()
image = pb.load_single_band(fname, np.uint8)

#               Detect QR codes in image
detector.detect(image)
print("Detected a total of {} QR Codes".format(len(detector.detections)))

for qr in detector.detections:
    print("Message: "+qr.message)
    print("     at: "+str(qr.bounds))

#               Construct 2D image points
object_points_1 = detector.detections[0].bounds
object_points_2 = detector.detections[1].bounds

A_imgpts = np.array([
    [object_points_1.vertexes[3].x, object_points_1.vertexes[3].y],
    [object_points_1.vertexes[2].x, object_points_1.vertexes[2].y],
    [object_points_1.vertexes[0].x, object_points_1.vertexes[0].y],
    [object_points_1.vertexes[1].x, object_points_1.vertexes[1].y]
], dtype=np.float64)

B_imgpts = np.array([
    [object_points_2.vertexes[3].x, object_points_2.vertexes[3].y],
    [object_points_2.vertexes[2].x, object_points_2.vertexes[2].y],
    [object_points_2.vertexes[0].x, object_points_2.vertexes[0].y],
    [object_points_2.vertexes[1].x, object_points_2.vertexes[1].y]
], dtype=np.float64)

retval, A_rvecs, A_tvecs = cv2.solvePnP(A_objpts, A_imgpts, mtx, dist)
retval, B_rvecs, B_tvecs = cv2.solvePnP(B_objpts, B_imgpts, mtx, dist)

img = cv2.imread(fname)
A_axis_imgpts, jac = cv2.projectPoints(axis, A_rvecs, A_tvecs, mtx, dist)
draw_3D_axis(img, (object_points_1.vertexes[3].x, object_points_1.vertexes[3].y), A_axis_imgpts)

B_axis_imgpts, jac = cv2.projectPoints(axis, B_rvecs, B_tvecs, mtx, dist)
draw_3D_axis(img, (object_points_2.vertexes[3].x, object_points_2.vertexes[3].y), B_axis_imgpts)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

A_angle_rad = np.sqrt((A_rvecs[0]*A_rvecs[0]) + (A_rvecs[1]*A_rvecs[1]) + (A_rvecs[2]*A_rvecs[2]))
B_angle_rad = np.sqrt((B_rvecs[0]*B_rvecs[0]) + (B_rvecs[1]*B_rvecs[1]) + (B_rvecs[2]*B_rvecs[2]))

#   use Rodrigues to transform pvr to dcm
AdcmO,_ = cv2.Rodrigues(src=A_rvecs)
BdcmO,_ = cv2.Rodrigues(src=B_rvecs)

#   cast to matrix to perform dot product
AdcmO = Matrix(AdcmO)
BdcmO = Matrix(BdcmO)

AdcmB1 = AdcmO*BdcmO.transpose()
BdcmA1 = BdcmO*AdcmO.transpose()

print("AdcmB1:", AdcmB1)
print("BdcmA1:", BdcmA1)

print("complete")