import numpy as np
import scipy.io
import os
import load_cnn

def init_features(features, **gparams, is_color_image = False, img_sample_sz = 0, size_mode = ''):
    if (size_mode == ''):
        size_mode = 'same'

    gp_keys = gparams.keys()
    if not 'normalize_power' in gp_keys:
        gparams['normalize_power'] = []
    if not 'normalize_size' in gp_keys:
        gparams['normalize_size'] = True
    if not 'normalize_dim' in gp_keys:
        gparams['normalize_dim'] = False
    if not 'square_root_normalization' in gp_keys:
        gparams['square_root_normalization'] = False
    if not 'use_gpu' in gp_keys:
        gparams['use_gpu'] = False

    keep_features = []
    for i in range(1:len(features)):
        f_keys = features[i]['fparams']
        if not 'useForColor' in f_keys:
            features[i]['fparams']['useForColor'] = True
        if not 'useForGray' in f_keys:
            features[i]['fparams']['useForGray'] = True

        if ((features[i]['fparams']['useForColor'] && is_color_image) ||
            (features[i]['fparams']['useForGray'] && !is_color_image)):
            keep_features.append(features[i])

    features = keep_features
    num_features = len(features)
    feature_info = {}
    feature_info['min_cell_size'] = np.zeros((num_frames, 1))

    for i in range(1:len(features)):
        if 'get_fhog' in features[i].keys():
            if not 'nOrients' in features[i]["fparams"].keys():
                features[i]["fparams"]["nOrients"] = 9
            features[i]["fparams"]["nDim"] = 3*features[i]["fparams"]["nOrients"] + 5 - 1
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False

        elif 'get_table_feature' in features[i]["fparams"].keys():
            cur_path = os.path.dirname(os.path.abspath(__file__))
            load_path = cur_path + '/lookup_tables' + features[i]["fparams"]["tablename"]
            table = scipy.io.loadmat(load_path)
            features[i]["fparams"]["nDim"] = table[features[i]["fparams"]["tablename"]].shape[1]
            features[i]["is_cell"] = False
            features[i]["ic_cnn"] = False

        elif 'get_colorspace' in features[i]["fparams"].keys():
            features[i]["fparams"]["nDim"] = 1
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False

        elif 'get_cnn_layers' in features[i]["fparams"].keys() || 'get_OFcnn_layers' in features[i]["fparams"].keys():
            features[i]["fparams"]["output_layer"].sort()
            if not 'input_size_mode' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_mode"] = 'adaptive'
            if not 'input_size_scale' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_scale"] = 1
            if not 'downsample_factor' in features[i]["fparams"].keys():
                features[i]["fparams"]["downsample_factor"] = np.ones((1, len(features[i]["fparams"]["output_layer"])))

            net = load_cnn(features[i]["fparams"], img_sample_sz)
            features[i]["fparams"]["nDim"] = "net[\"info\"].dataSize[2:features{k}.fparams.output_layer+1]"

            if 'receptiveFieldStride' in net["info"].keys():
                shape = net["info"].receptiveFieldStride.shape
                net_info_stride = np.ones(sape[0], shape[1]+1)
                net_info_stride[:,1:shape[1]+1] = net["info"].receptiveFieldStride
            else:
                net_info_stride = np.array([[1],[1]])

            for layer in features[i]["fparams"]["output_layer"]:
                net_stride.append(net_info_stride[0][layer + 1])
            net_stride = net_stride[np.newaxis]
            net_stride = net_stride
            downsample_factor = features[i]["fparams"]["downsample_factor"][np.newaxis]
            downsample_factor = downsample_factor.T
            features[i]["fparams"]["cell_size"] = net_stride * downsample_factor

            features[i]["is_cell"] = True
            features[i]["is_cnn"] = True
        else:
            raise ValueError("Unknown feature type")

        if not 'cell_size' in features[i]["fparams"].keys():
            features[i]["fparams"]["cell_size"] = 1
        if not 'penalty' in features[i]["fparams"].keys():
            features[i]["fparams"]["penalty"] = np.zeros((len(features[i]["fparams"]["nDim"]), 1))
        feature_info["min_cell_size"][i] = min(features[i]["fparams"]["cell_size"])

    
