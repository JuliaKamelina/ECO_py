import mxnet as mx
import numpy as np
import cv2 as cv

from mxnet.gluon.model_zoo import vision
from mxnet.gluon.nn import AvgPool2D

from . import py_gradient

def feature_normalization(x, gparams):
    if ('normalize_power' in gparams.keys()) and gparams["normalize_power"] > 0:
        if gparams["normalize_power"] == 2:
            x = x * np.sqrt((x.shape[0]*x.shape[1]) ** gparams["normalize_size"] * (x.shape[2]**gparams["normalize_dim"]) / (x**2).sum(axis=(0, 1, 2)))
        else:
            x = x * ((x.shape[0]*x.shape[1]) ** gparams["normalize_size"]) * (x.shape[2]**gparams["normalize_dim"]) / ((np.abs(x) ** (1. / gparams["normalize_power"])).sum(axis=(0, 1, 2)))

    if gparams["square_root_normalization"]:
        x = np.sign(x) * np.sqrt(np.abs(x))
    return x.astype(np.float32)

def get_sample(im, pos, img_sample_sz, output_sz):
    pos = np.floor(pos)
    sample_sz = np.maximum(np.round(img_sample_sz), 1)
    x = np.floor(pos[1]) + np.arange(0, img_sample_sz[1]+1) - np.floor((img_sample_sz[1]+1)/2)
    y = np.floor(pos[0]) + np.arange(0, img_sample_sz[0]+1) - np.floor((img_sample_sz[0]+1)/2)
    x_min = max(0, int(x.min()))
    x_max = min(im.shape[1], int(x.max()))
    y_min = max(0, int(y.min()))
    y_max = min(im.shape[0], int(y.max()))
    # extract image
    im_patch = im[y_min:y_max, x_min:x_max, :]
    left = right = top = down = 0
    if x.min() < 0:
        left = int(abs(x.min()))
    if x.max() > im.shape[1]:
        right = int(xs.max() - im.shape[1])
    if y.min() < 0:
        top = int(abs(y.min()))
    if y.max() > im.shape[0]:
        down = int(y.max() - im.shape[0])
    if left != 0 or right != 0 or top != 0 or down != 0:
        im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
    im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv2.INTER_CUBIC)
    if len(im_patch.shape) == 2:
        im_patch = im_patch[:, :, np.newaxis]
    return im_patch

def forvard_pass(x):
    vgg16 = vision.vgg16(pretrained=True)
    avg_pool2d = AvgPool2D()

    conv1_1 = vgg16.features[0].forward(x)
    relu1_1 = vgg16.features[1].forward(conv1_1)
    conv1_2 = vgg16.features[2].forward(relu1_1)
    relu1_2 = vgg16.features[3].forward(conv1_2)
    pool1 = vgg16.features[4].forward(relu1_2) # x2
    pool_avg = avg_pool2d(pool1)

    conv2_1 = vgg16.features[5].forward(pool1)
    relu2_1 = vgg16.features[6].forward(conv2_1)
    conv2_2 = vgg16.features[7].forward(relu2_1)
    relu2_2 = vgg16.features[8].forward(conv2_2)
    pool2 = vgg16.features[9].forward(relu2_2) # x4

    conv3_1 = vgg16.features[10].forward(pool2)
    relu3_1 = vgg16.features[11].forward(conv3_1)
    conv3_2 = vgg16.features[12].forward(relu3_1)
    relu3_2 = vgg16.features[13].forward(conv3_2)
    conv3_3 = vgg16.features[14].forward(relu3_2)
    relu3_3 = vgg16.features[15].forward(conv3_3)
    pool3 = vgg16.features[16].forward(relu3_3) # x8

    conv4_1 = vgg16.features[17].forward(pool3)
    relu4_1 = vgg16.features[18].forward(conv4_1)
    conv4_2 = vgg16.features[19].forward(relu4_1)
    relu4_2 = vgg16.features[20].forward(conv4_2)
    conv4_3 = vgg16.features[21].forward(relu4_2)
    relu4_3 = vgg16.features[22].forward(conv4_3)
    pool4 = vgg16.features[23].forward(relu4_3) # x16
    return [pool_avg.asnumpy().transpose(2, 3, 1, 0),
            pool4.asnumpy().transpose(2, 3, 1, 0)]

def get_cnn_layers(im, fparams, gparams, pos, sample_sz, scale_factor):
    compressed_dim = fparams["compressed_dim"] # TODO: check
    cell_size = fparams["cell_size"]
    penalty = fparams["penalty"]
    min_cell_size = np.min(cell_size)

    if im.shape[2] == 1:
        im = cv2.cvtColor(im.squeeze(), cv2.COLOR_GRAY2RGB)
    if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
        scale_factor = [scale_factor]
    patches = []
    for scale in scale_factor:
        patch = get_sample(im, pos, sample_sz*scale_factor, sample_sz)
        patch = mx.nd.array(patch / 255.)
        normalized = mx.image.color_normalize(patch, mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                    std=mx.nd.array([0.229, 0.224, 0.225]))
        normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
    patches.append(normalized)
    patches = mx.nd.concat(*patches, dim=0)
    f1, f2 = forward_pass(patches)
    f1 = feature_normalization(f1)
    f2 = feature_normalization(f2)
    return f1, f2

    def get_fhog(img, fparams, gparam, pos, sample_sz, scales):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = get_sample(img, pos, sample_sz*scale, sample_sz)
            # h, w, c = patch.shape
            M, O = py_gradient.gradMag(patch.astype(np.float32), 0, True)
            H = py_gradient.fhog(M, O, fparams["cell_size"], fparams["nOrients"], -1, .2)
            # drop the last dimension
            H = H[:, :, :-1]
            feat.append(H)
        feat = feature_normalization(np.stack(feat, axis=3), gparams)
        return [feat]
    }
