import cv2
import pyqrcode
import qrcodes
import draw_axis
import create_background
from PIL import Image
import GetVectors
import triad2D
import numpy as np

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
rotated_qr_image.save("rot_pill_image.png")
rotated_qr_image2.save("rot_pill_image2.png")
background.save("background_qrcodes.png")

background_cv_image = cv2.imread("background_qrcodes.png")
rot_cv_image = cv2.imread("rot_pill_image.png")
rot_cv_image2 = cv2.imread("rot_pill_image2.png")

#               Detect QR-codes in images
decoded_objects_1 = qrcodes.decode(rot_cv_image)
decoded_objects_2 = qrcodes.decode(background_cv_image)
decoded_objects_3 = qrcodes.decode(rotated_qr_image2)

#               Draw axis on images
object_points_1 = qrcodes.get_points(rot_cv_image, decoded_objects_1)
draw_axis.draw_axis(rot_cv_image, object_points_1)

object_points_2 = qrcodes.get_points(background_cv_image, decoded_objects_2)
draw_axis.draw_axis(background_cv_image, object_points_2)

object_points_3 = qrcodes.get_points(rot_cv_image2, decoded_objects_3)
draw_axis.draw_axis(rot_cv_image2, object_points_3)

#               Display the ouput image
cv2.imshow('image', rot_cv_image)
cv2.imshow('backgournd', background_cv_image)
cv2.imshow('image2', rot_cv_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#               Send QR-corner points to GetCoords function
Q1 = np.array([
    [object_points_2[1][1][0], object_points_2[1][1][1], 0],
    [object_points_2[1][0][0], object_points_2[1][0][1], 0],
    [object_points_2[1][2][0], object_points_2[1][2][1], 0],
    [object_points_2[1][1][0], object_points_2[1][1][1], 1]
])

Q2 = np.array([
    [object_points_2[0][1][0], object_points_2[0][1][1], 0],
    [object_points_2[0][0][0], object_points_2[0][0][1], 0],
    [object_points_2[0][2][0], object_points_2[0][2][1], 0],
    [object_points_2[0][1][0], object_points_2[0][1][1], 1]
])

A, B = GetCoords.get_coords(Q1, Q2)
AdcmB, trueAdcmB = triad2D.triad2d(A, B)






