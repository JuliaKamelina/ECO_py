import numpy as np
import scipy.io

def read_cnn(net):
    input_size = []
    if ('inputSize' in net["meta"]._fieldnames):
        input_size = net["meta"].inputSize
    if ('normalization' in net["meta"]._fieldnames and
        'imageSize' in net["meta"].normalization._fieldnames):
        input_size = net["meta"].normalization.imageSize

    info = {}
    info["support"] = np.array([])
    info["stride"] = np.array([])
    info["pad"] = np.array([])
    info["receptiveFieldSize"] = []
    info["receptiveFieldOffset"] = []
    info["receptiveFieldStride"] = []

    for i in range(0, len(net["layers"])):
        layer = net["layers"][i]
        support = []
        if (layer.type == "conv"):
            support.append(max(layer.weights[0].shape[0], 1))
            support.append(max(layer.weights[0].shape[1], 1))
            support = np.array(support)
            support = (support - 1)*layer.dilate + 1
            info["support"] = np.insert(info["support"], i, support, axis=0)
        elif (layer.type == "pool"):
            info["support"] = np.insert(info["support"], i, np.array(layer.pool), axis=0)
        else:
            info["support"] = np.insert(info["support"], i, np.array([1,1]), axis=0)

        if (len(info["support"].shape) == 1):
            info["support"] = info["support"].reshape(1, info["support"].shape[0])

        if ('stride' in layer._fieldnames):
            info["stride"] = np.insert(info["stride"], i, np.array(layer.stride), axis=0)
        else:
            info["stride"] = np.insert(info["stride"], i, np.array([1,1]), axis=0)

        if (len(info["stride"].shape) == 1):
            info["stride"] = info["stride"].reshape(1, info["stride"].shape[0])

        if ('pad' in layer._fieldnames):
            info["pad"] = np.insert(info["pad"], i, np.array(layer.pad), axis=0)
        else:
            info["pad"] = np.insert(info["pad"], i, np.array([0, 0, 0, 0]), axis=0)

        if (len(info["pad"].shape) == 1):
            info["pad"] = info["pad"].reshape(1, info["pad"].shape[0])

        tmp_stride = []
        tmp_support = info["support"][0]
        tmp_pad = info["pad"][0, ::2]
        if (i > 1):
            tmp_stride = info["stride"][0:i]
            tmp_support = info["support"][0:i+1]
            tmp_pad = info["pad"][0:i+1, ::2]
        elif (i == 1):
            tmp_stride = info["stride"][0]
            tmp_support = info["support"][0:i+1]
            tmp_pad = info["pad"][0:i+1, ::2]

        if (len(tmp_stride) != 0 and len(tmp_stride.shape) == 1):
            tmp_stride = tmp_stride.reshape(1, tmp_stride.shape[0])
        tmp_stride = np.insert(tmp_stride, 0, [1, 1], axis=0)
        if (len(tmp_stride.shape) == 1):
            tmp_stride = tmp_stride.reshape(1, tmp_stride.shape[0])
        # print(tmp_support)
        # print(np.sum(np.cumprod(tmp_stride, axis=0)*
                                              # (tmp_support - 1), axis=0))
        info["receptiveFieldSize"].append(1 + np.sum(np.cumprod(tmp_stride, axis=0)*
                                              (tmp_support - 1), axis=0))
        info["receptiveFieldOffset"].append(1 + np.sum(np.cumprod(tmp_stride, axis=0)*
                                                ((tmp_support-1)/2 - tmp_pad), axis=0))
        info["receptiveFieldStride"] = np.cumprod(info["stride"], axis=0)

    info["receptiveFieldSize"] = np.array(info["receptiveFieldSize"])
    info["receptiveFieldOffset"] = np.array(info["receptiveFieldOffset"])
    info["receptiveFieldStride"] = np.array(info["receptiveFieldStride"])

    info["dataSize"] = np.array(input_size)
    info["dataSize"] = info["dataSize"].reshape(1,len(input_size))

    for i in range(0, len(net["layers"])):
        layer = net["layers"][i]
        data_size = []
        data_size.append(np.floor((info["dataSize"][i, 0] +
                                  np.sum(info["pad"][i, 0:2]) -
                                  info["support"][i,0])/info["stride"][i,0]) + 1)
        data_size.append(np.floor((info["dataSize"][i,1] +
                                  np.sum(info["pad"][i, 2:4]) -
                                  info["support"][i,1])/info["stride"][i,1]) + 1)
        data_size.append(info["dataSize"][i,2])
        print(info["dataSize"][i,2])
        data_size.append(info["dataSize"][i,3])

        if (layer.type == "conv"):
            f = []
            if ("weights" in layer._fieldnames):
                f = layer.weights[0]
            else:
                f = layer.filters
            if (len(f.shape) > 3 and f.shape[2] != 0):
                data_size[2] = f.shape[3]
            elif(len(f.shape) <= 3):
                data_size[2] = 1

        info["dataSize"] = np.insert(info["dataSize"], i+1, data_size, axis=0)

    return (info)

if __name__ == "__main__":
    net = scipy.io.loadmat('C:/Users/jkamelin/Documents/cw/ECO_py/feature_extraction/networks/imagenet-vgg-m-2048.mat', squeeze_me=True, struct_as_record=False)
    info = read_cnn(net)
    print(info)
