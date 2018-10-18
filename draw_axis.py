import matplotlib.axes
import cv2
import qrcodes


def draw_axis(im, objects):
    for obj in objects:
        top_left = obj[0]
        bot_left = obj[1]
        bot_rigth = obj[2]

        cv2.line(im, bot_left, top_left, (255, 0, 0), 2)
        cv2.line(im, bot_left, bot_rigth, (0, 0, 255), 2)


