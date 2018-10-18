from PIL import Image



def create_background(width, height):
    background = Image.new(mode='L', size=(width, height), color=255)
    return background

