import cv2
import numpy as np


class CheckerBoard:
    blocksize = (20, 20)
    imagesize = (160, 160)
    chessboard = 0

    def draw_square(self, src, size, str_point, clr):
        """
        :param src: the image to draw upon
        :param size: (height,width)/(rows,cols) of block to be drawn
        :param str_point: the top left cornes of sqaure (horisontal displacement,vertical displacement)
        :param clr: the color of the sqaure
        :return: none
        """
        # clr = np.uint8(np.random.uniform(0, 255, 3))
        # c = tuple(map(int, clr))
        clr = tuple(map(int, clr))
        cv2.rectangle(img=src, pt1=str_point, pt2=(str_point[0] + size[1], str_point[1] + size[0]), color=clr,
                      thickness=-1)
        print("rectangle drawn")
        return

    def draw_blankimage(self, size):
        """draws a blank imagge to draw upon
        :rtype: object
        :param size: the size of the image to be created (height,width)/(rows,cols)
        :return: an image(array of empty piont) of size specified
        """
        dst = np.zeros(size, np.uint8)
        return dst

    def draw_chessboard(self):
        """draws a checker board of size imagesize"""

        color = np.uint8(255)
        blankimage = self.draw_blankimage(self.imagesize)
        for x in range(0, 8):
            color = ~color
            for y in range(0, 8):
                x_str = x * self.blocksize[1]  # blocksize = (width,height)
                y_str = y * self.blocksize[0]
                self.draw_square(blankimage, self.blocksize, (x_str, y_str), (color, color, color))
                color = ~color
        self.chessboard = blankimage
        return
