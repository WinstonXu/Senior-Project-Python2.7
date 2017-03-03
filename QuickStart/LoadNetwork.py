from keras.models import load_model
from PIL import Image
import os
import numpy as np
import sys

def testmodel(file_name):
    print os.path.dirname(__file__)
    os.chdir(os.path.curdir)
    model = load_model('pic_model.h5')
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "Uploads", file_name)):
        print "Image does not exist"
    else:
        im = Image.open(os.path.join(os.path.dirname(__file__), "Uploads", file_name))
        im = im.resize((84,28))
        im = im.convert('L')
        # newim = ImageEnhance.Contrast(im)
        # newim = newim.enhance(2.5).save()
        pixelData = im.getdata()
        pixelData = np.asarray(pixelData, dtype=np.float64).reshape((1, im.size[0], im.size[1], 1))
        pixelData *= 1 / 255.
        res = model.predict(pixelData)
        # print res.shape
        equation = ""
        equation += "%d" %(np.argmax(res[0][:10]))
        operator = np.argmax(res[0][10:14])
        if operator == 0:
            equation += "+"
        if operator == 1:
            equation += "-"
        if operator == 2:
            equation += "x"
        if operator == 3:
            equation += "/"
        equation += "%d" %(np.argmax(res[0][14:]))
        return equation

if __name__ == "__main__":
    testmodel(sys.argv[1])