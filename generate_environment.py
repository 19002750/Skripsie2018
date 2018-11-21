import create_background
from PIL import Image
from generate_qrcode import generateQRcodes
import numpy as np


class Environment():
    num_qr = 0
    qr_rotations = []

    def generate_env(self, num_qr, qr_rotations, env_filename):
        background = create_background.create_background(800, 800)
        self.num_qr = num_qr
        self.qr_rotations = qr_rotations
        qr_num = 1
        position = 0
        for qr_rot in qr_rotations:
            #               Get QR-code
            QR = generateQRcodes()
            qr_filename = "Q"+str(qr_num)+".png"
            qr_image = QR.generate_qr(qr_version=2, qr_scale=2, qr_data="Q"+str(qr_num), qr_filename=qr_filename)
            #               Get rotation image
            basewidth = 200
            wpercent = (basewidth / float(qr_image.size[0]))
            hsize = int(float(qr_image.size[1]) * float(wpercent))
            rotated_qr_image = qr_image.resize((basewidth, hsize), Image.ANTIALIAS).rotate(qr_rot, expand=True)
            #               Add qrcodes to background
            background.paste(rotated_qr_image, (position*250+50, position*250+50))
            qr_num = qr_num +1
            position = position +1

        background.save(env_filename)
