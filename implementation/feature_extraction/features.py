import mxnet as mx
import numpy as np
import cv2 as cv
import scipy.io
import os

from mxnet.gluon.model_zoo import vision
from mxnet.gluon.nn import AvgPool2D
from ._gradient import *

from ..runfiles import settings

class Features():
    def __init__(self, is_color=False):
        self.is_color_image = is_color

    def _round(self, x):
        res = x.copy()
        res[0] = np.ceil(x[0]) if x[0] - np.floor(x[0]) >= 0.5 else np.floor(x[0])
        res[1] = np.ceil(x[1]) if x[1] - np.floor(x[1]) >= 0.5 else np.floor(x[1])
        return res

    # @staticmethod
    def _set_size(self, img_sample_sz, size_mode):
        new_img_sample_sz = np.array(img_sample_sz, dtype=np.int32)
        if size_mode != "same" and settings.cnn_params['input_size_mode'] == "adaptive":
            orig_sz = np.ceil(new_img_sample_sz/16)

            if size_mode == "exact":
                desired_sz = orig_sz + 1
            elif size_mode == "odd_cells":
                desired_sz = orig_sz + 1 + orig_sz%2
            new_img_sample_sz = desired_sz*16

        if settings.cnn_params['input_size_mode'] == "adaptive":
            img_sample_sz = np.round(new_img_sample_sz) # == feature_info.img_support_sz
        else:
            img_sample_sz = np.array(img_sample_sz) #net["meta"].normalization.imageSize[0:2]
        return img_sample_sz

    def get_feature(self, im, pos, sample_sz, scale_factor):
        pass

    def feature_normalization(self, x):
        gparams = settings.t_global
        if gparams["normalize_power"] > 0:
            if gparams["normalize_power"] == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** gparams["normalize_size"] * (x.shape[2]**gparams["normalize_dim"]) / (x**2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** gparams["normalize_size"]) * (x.shape[2]**gparams["normalize_dim"]) / ((np.abs(x) ** (1. / gparams["normalize_power"])).sum(axis=(0, 1, 2)))

        if gparams["square_root_normalization"]:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)

    def get_sample(self, im, pos, img_sample_sz, output_sz):
        pos = np.floor(pos)
        sample_sz = np.maximum(self._round(img_sample_sz), 1)
        x = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
        y = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
        x_min = max(0, int(x.min()))
        x_max = min(im.shape[1], int(x.max()))
        y_min = max(0, int(y.min()))
        y_max = min(im.shape[0], int(y.max()))

        if len(im) == 3:
            im_patch = im[y_min:y_max, x_min:x_max, :]
        else:
            im_patch = im[y_min:y_max, x_min:x_max]
        left = right = top = down = 0

        if x.min() < 0:
            left = int(abs(x.min()))
        if x.max() > im.shape[1]:
            right = int(x.max() - im.shape[1])

        if y.min() < 0:
            top = int(abs(y.min()))
        if y.max() > im.shape[0]:
            down = int(y.max() - im.shape[0])

        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv.copyMakeBorder(im_patch, top, down, left, right, cv.BORDER_REPLICATE)
        im_patch = cv.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv.INTER_CUBIC)

        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch


class Resnet18Features(Features):
    def __init__(self, is_color, img_sample_sz=[], size_mode='same'):
        super().__init__(is_color)
        use_for_color = settings.cnn_params.get('useForColor', True)
        use_for_gray = settings.cnn_params.get('useForGray', True)
        self.use_feature = (use_for_color and is_color) or (use_for_gray and not is_color)

        self.net = vision.resnet18_v2(pretrained=True, ctx = mx.cpu(0))
        self.compressed_dim = settings.cnn_params['compressed_dim']
        self.cell_size = np.array([4, 16])
        self.penalty = np.zeros((2, 1))
        self.nDim = np.array([64, 256])
        self.img_sample_sz = self._set_size(img_sample_sz, size_mode)
        self.data_sz = np.ceil(self.img_sample_sz / self.cell_size[:, None])

    def forward_pass(self, x):
        bn0 = self.net.features[0].forward(x)
        conv1 = self.net.features[1].forward(bn0)     # x2
        bn1 = self.net.features[2].forward(conv1)
        relu1 = self.net.features[3].forward(bn1)
        pool1 = self.net.features[4].forward(relu1)   # x4
        # stage1
        stage1 = self.net.features[5].forward(pool1)  # x4
        stage2 = self.net.features[6].forward(stage1) # x8
        stage3 = self.net.features[7].forward(stage2) # x16
        return [pool1.asnumpy().transpose(2, 3, 1, 0),
                stage3.asnumpy().transpose(2, 3, 1, 0)]

    def get_feature(self, im, pos, sample_sz, scale_factor):
        if len(im.shape) == 2:
            im = cv.cvtColor(im.squeeze(), cv.COLOR_GRAY2RGB)
        if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
            scale_factor = [scale_factor]
        patches = []
        for scale in scale_factor:
            patch = self.get_sample(im, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch / 255.)
            normalized = mx.image.color_normalize(patch, mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                        std=mx.nd.array([0.229, 0.224, 0.225]))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
        patches = mx.nd.concat(*patches, dim=0)
        f1, f2 = self.forward_pass(patches)
        f1 = self.feature_normalization(f1)
        f2 = self.feature_normalization(f2)
        return f1, f2


class Resnet50Features(Features):
    def __init__(self, is_color, img_sample_sz=[], size_mode='same'):
        super().__init__(is_color)
        use_for_color = settings.cnn_params.get('useForColor', True)
        use_for_gray = settings.cnn_params.get('useForGray', True)
        self.use_feature = (use_for_color and is_color) or (use_for_gray and not is_color)

        self.net = vision.resnet50_v2(pretrained=True, ctx = mx.cpu(0))
        self.compressed_dim = settings.cnn_params['compressed_dim']
        self.cell_size = np.array([4, 16])
        self.penalty = np.zeros((2, 1))
        self.nDim = np.array([64, 1024])
        self.img_sample_sz = self._set_size(img_sample_sz, size_mode)
        self.data_sz = np.ceil(self.img_sample_sz / self.cell_size[:, None])

    def forward_pass(self, x):
        bn0 = self.net.features[0].forward(x)
        conv1 = self.net.features[1].forward(bn0)     # x2
        bn1 = self.net.features[2].forward(conv1)
        relu1 = self.net.features[3].forward(bn1)
        pool1 = self.net.features[4].forward(relu1)   # x4
        # stage1
        stage1 = self.net.features[5].forward(pool1)  # x4
        stage2 = self.net.features[6].forward(stage1) # x8
        stage3 = self.net.features[7].forward(stage2) # x16
        return [pool1.asnumpy().transpose(2, 3, 1, 0),
                stage3.asnumpy().transpose(2, 3, 1, 0)]

    def get_feature(self, im, pos, sample_sz, scale_factor):
        if len(im.shape) == 2:
            im = cv.cvtColor(im.squeeze(), cv.COLOR_GRAY2RGB)
        if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
            scale_factor = [scale_factor]
        patches = []
        for scale in scale_factor:
            patch = self.get_sample(im, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch / 255.)
            normalized = mx.image.color_normalize(patch, mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                        std=mx.nd.array([0.229, 0.224, 0.225]))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
        patches = mx.nd.concat(*patches, dim=0)
        f1, f2 = self.forward_pass(patches)
        f1 = self.feature_normalization(f1)
        f2 = self.feature_normalization(f2)
        return f1, f2


class VGGFeatures(Features):
    def __init__(self, is_color, img_sample_sz=[], size_mode='same'):
        super().__init__(is_color)
        use_for_color = settings.cnn_params.get('useForColor', True)
        use_for_gray = settings.cnn_params.get('useForGray', True)

        self.net = vision.vgg16(pretrained=True)
        self.pool2d = AvgPool2D()

        self.use_feature = (use_for_color and is_color) or (use_for_gray and not is_color)
        self.nDim = np.array([64, 512]) #[96 512] net["info"]["dataSize"][layer_dim_ind, 2]
        self.cell_size = np.array([4, 16])
        self.penalty = np.zeros((2, 1))
        self.compressed_dim = settings.cnn_params['compressed_dim']
        self.img_sample_sz = self._set_size(img_sample_sz, size_mode)
        self.data_sz = np.ceil(self.img_sample_sz / self.cell_size[:, None])

    def forward_pass(self, x):
        conv1_1 = self.net.features[0].forward(x)
        relu1_1 = self.net.features[1].forward(conv1_1)
        conv1_2 = self.net.features[2].forward(relu1_1)
        relu1_2 = self.net.features[3].forward(conv1_2)
        pool1 = self.net.features[4].forward(relu1_2) # x2
        pool_avg = self.pool2d(pool1)

        conv2_1 = self.net.features[5].forward(pool1)
        relu2_1 = self.net.features[6].forward(conv2_1)
        conv2_2 = self.net.features[7].forward(relu2_1)
        relu2_2 = self.net.features[8].forward(conv2_2)
        pool2 = self.net.features[9].forward(relu2_2) # x4

        conv3_1 = self.net.features[10].forward(pool2)
        relu3_1 = self.net.features[11].forward(conv3_1)
        conv3_2 = self.net.features[12].forward(relu3_1)
        relu3_2 = self.net.features[13].forward(conv3_2)
        conv3_3 = self.net.features[14].forward(relu3_2)
        relu3_3 = self.net.features[15].forward(conv3_3)
        pool3 = self.net.features[16].forward(relu3_3) # x8

        conv4_1 = self.net.features[17].forward(pool3)
        relu4_1 = self.net.features[18].forward(conv4_1)
        conv4_2 = self.net.features[19].forward(relu4_1)
        relu4_2 = self.net.features[20].forward(conv4_2)
        conv4_3 = self.net.features[21].forward(relu4_2)
        relu4_3 = self.net.features[22].forward(conv4_3)
        pool4 = self.net.features[23].forward(relu4_3) # x16
        return [pool_avg.asnumpy().transpose(2, 3, 1, 0),
                pool4.asnumpy().transpose(2, 3, 1, 0)]

    def get_feature(self, im, pos, sample_sz, scale_factor):
        if len(im.shape) == 2:
            im = cv.cvtColor(im.squeeze(), cv.COLOR_GRAY2RGB)
        if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
            scale_factor = [scale_factor]
        patches = []
        for scale in scale_factor:
            patch = self.get_sample(im, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch / 255.)
            normalized = mx.image.color_normalize(patch, mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                        std=mx.nd.array([0.229, 0.224, 0.225]))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
        patches = mx.nd.concat(*patches, dim=0)
        f1, f2 = self.forward_pass(patches)
        f1 = self.feature_normalization(f1)
        f2 = self.feature_normalization(f2)
        return f1, f2


class HOGFeatures(Features):
    def __init__(self, is_color, img_sample_sz=[], size_mode='same'):
        super().__init__(is_color)
        use_for_color = settings.hog_params.get('useForColor', True)
        use_for_gray = settings.hog_params.get('useForGray', True)
        self.use_feature = (use_for_color and is_color) or (use_for_gray and not is_color)
        self.nOrients = settings.hog_params.get('nOrients', 9)
        self.nDim = np.array([3*self.nOrients + 5 - 1])
        self.compressed_dim = settings.hog_params['compressed_dim']
        self.img_sample_sz = self._set_size(img_sample_sz, size_mode)
        self.cell_size = settings.hog_params.get('cell_size')
        self.data_sz = np.ceil(self.img_sample_sz / self.cell_size)

    def get_feature(self, img, pos, sample_sz, scale_factor):
        feat = []
        if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
            scale_factor = [scale_factor]
        for scale in scale_factor:
            patch = self.get_sample(img, pos, sample_sz*scale, sample_sz)
            # h, w, c = patch.shape
            M, O = gradMag(patch.astype(np.float32), 0, True)
            H = fhog(M, O, self.cell_size, self.nOrients, -1, .2)
            # drop the last dimension
            H = H[:, :, :-1]
            feat.append(H)
        feat = self.feature_normalization(np.stack(feat, axis=3))
        return [feat]


class TableFeatures(Features):
    def __init__(self, is_color, img_sample_sz=[], size_mode='same'):
        super().__init__(is_color)
        use_for_color = settings.cn_params.get('useForColor', True)
        use_for_gray = settings.cn_params.get('useForGray', True)
        self.use_feature = (use_for_color and is_color) or (use_for_gray and not is_color)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        load_path = cur_path + '/lookup_tables/' + settings.cn_params.get("tablename")
        self.table = scipy.io.loadmat(load_path)
        self.compressed_dim = settings.cn_params['compressed_dim']
        self.img_sample_sz = self._set_size(img_sample_sz, size_mode)
        self.cell_size = settings.hog_params.get('cell_size')
        self.data_sz = np.ceil(self.img_sample_sz / self.cell_size)

    @staticmethod
    def integralImage(img):
        w, h, c = img.shape
        intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        return intImage

    @staticmethod
    def avg_feature_region(features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = TableFeatures.integralImage(features)
        i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
        return region_image

    def get_feature(self, img, pos, sample_sz, scale_factor):
        feat = []
        factor = 32
        den = 8

        if not isinstance(scale_factor, list) and not isinstance(scale_factor, np.ndarray):
            scale_factor = [scale_factor]
        for scale in scale_factor:
            patch = self.get_sample(img, pos, sample_sz*scale, sample_sz)
            h, w, c = patch.shape
            if c == 3:
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // den + (GG // den) * factor + (BB // den) * factor * factor
                f = self.table[index.flatten()].reshape((h, w, self.table.shape[1]))
            else:
                f = self.table[patch.flatten()].reshape((h, w, self.table.shape[1]))
            if settings.cn_params["cell_size"] > 1:
                f = self.avg_feature_region(f, settings.cn_params["cell_size"])
            feat.append(f)
        feat = self.feature_normalization(np.stack(feat, axis=3))
        return [feat]


def _fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = gradMag(I.astype(np.float32), 0, True)
    H = fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H