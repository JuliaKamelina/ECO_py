import scipy.io
import os
import cv2 as cv
import numpy as np
from read_cnn import *

def load_cnn(fparams, im_size):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    load_path = cur_path + '/networks' + fparams["nn_name"]
    net = scipy.io.loadmat(load_path, squeeze_me=True, struct_as_record=False)
    #vl_simplenn_tidy(net)
    net["layers"] = net["layers"][0:max(fparams["output_layer"])]

    if (fparams["input_size_mode"] == 'cnn_default'):
        base_input_sz = net["meta"].normalization.imageSize[0:2]
    elif (fparams["input_size_mode"] == 'adaptive'):
        base_input_sz = im_size[0:2]
    else:
        raise ValueError("Unnown input_size_mode")

    net["meta"].normalization.imageSize[0:2] = round(base_input_sz * fparams["input_size_scale"])
    #net["meta"].normalization.averageImageOri = net["meta"]["normalization"]["averageImage"]

    if ('inputSize' in net["meta"].keys()):
        net["meta"].inputSize = base_input_sz

    if (net["meta"].normalization.averageImage.shape[0] > 1 or net["meta"].normalization.averageImage.shape[1] > 1):
        average_image = np.array(net["meta"].normalization.averageImage).astype('float32')
        new_shape = tuple(net["meta"].normalization.imageSize[0:2])
        net["meta"].normalization.averageImage = cv.resize(average_image, new_shape)

    net["info"] = read_cnn(net)
    return (net)
