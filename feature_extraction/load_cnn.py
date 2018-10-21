import scipy.io
import os
import cv2 as cv

def load_cnn(**fparams, im_size):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    load_path = cur_path + '/networks' + faprams["nn_name"]
    net = scipy.io.loadmat(load_path)
    #vl_simplenn_tidy(net)
    net["layers"] = net["layers"][0:max(fparams["output_layer"])]

    if (fparams["input_size_mode"] == 'cnn_default'):
        base_input_sz = net["meta"]["normalization"]["imageSize"][0:2]
    elif (fparams["input_size_mode"] == 'adaptive'):
        base_input_sz = im_size[0:2]
    else:
        raise ValueError("Unnown input_size_mode")

    net["meta"]["normalization"]["imageSize"][0:2] = round(base_input_sz * fparams["input_size_scale"])
    net["meta"]["normalization"]["averageImageOri"] = net["meta"]["normalization"]["averageImage"]

    if ('inputSize' in net["meta"].keys()):
        net["meta"]["inputSize"] = base_input_sz

    if (net["meta"]["normalization"]["imageSize"].shape[0] > 1 || (net["meta"]["normalization"]["imageSize"].shape[1] > 1):
        net["meta"]["normalization"]["imageSize"] = cv.
