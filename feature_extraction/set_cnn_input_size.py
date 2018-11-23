import numpy as np
import cv2 as cv
import read_cnn

def set_cnn_input_size(net, im_size):
    net["meta"].normalization.imageSize[0:2] = np.round(im_size[0:2])
    net["meta"].normalization.averageImage = cv.imresize(net["meta"].normalization.averageImageOrig.astype('float32'), net["meta"].normalization.imageSize[0:2])

    net["info"] = read_cnn(net)
    return(net)