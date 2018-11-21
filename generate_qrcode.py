import pyqrcode
from PIL import Image


class generateQRcodes():

    qr_version = 0
    qr_scale = 0
    qr_data = "default"
    qr_filename = ""

    def generate_qr(self, qr_version, qr_scale, qr_data, qr_filename):
        #               Generate QR-Codes
        self.qr_version = qr_version
        self.qr_scale = qr_scale
        self.qr_data = qr_data
        self.qr_filename = qr_filename
        #               make instance of QRCode Class
        qr_object = pyqrcode.create(qr_data, error='H', version=qr_version)

        #               Make output image
        qr_img = qr_filename
        qr_object.png(qr_img, scale=qr_scale)
        pill_image = Image.open(qr_img)
        return pill_image

