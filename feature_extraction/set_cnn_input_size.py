import numpy as np
import cv2 as cv
from read_cnn import *

def set_cnn_input_size(net, im_size):
    net["meta"].normalization.imageSize[0:2] = np.round(im_size[0:2])
    net["meta"].normalization.averageImage = cv.resize(net["meta"].normalization.averageImageOrig.astype('float32'), tuple(net["meta"].normalization.imageSize[0:2]))

    net["info"] = read_cnn(net)
    return(net)