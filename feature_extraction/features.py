import mxnet as mx
import numpy as np
import cv2 as cv

from mxnet.gluon.model_zoo import vision
from mxnet.gluon.nn import AvgPool2D

def feature_normalization(x, gparams):
    if ('normalize_power' in gparams.keys()) and gparams["normalize_power"] > 0:
        if gparams["normalize_power"] == 2:
            x = x * np.sqrt((x.shape[0]*x.shape[1]) ** gparams["normalize_size"] * (x.shape[2]**gparams["normalize_dim"]) / (x**2).sum(axis=(0, 1, 2)))
        else:
            x = x * ((x.shape[0]*x.shape[1]) ** gparams["normalize_size"]) * (x.shape[2]**gparams["normalize_dim"]) / ((np.abs(x) ** (1. / gparams["normalize_power"])).sum(axis=(0, 1, 2)))

    if gparams["square_root_normalization"]:
        x = np.sign(x) * np.sqrt(np.abs(x))
    return x.astype(np.float32)

def get_cnn_layers(im, fparams, gparams):
    
