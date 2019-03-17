import numpy as np
import scipy.io
import os

def init_features(features, gparams, is_color_image = False, img_sample_sz = [], size_mode = ''):
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
    for i in range(0,len(features)):
        f_keys = features[i]['fparams']
        if not 'useForColor' in f_keys:
            features[i]['fparams']['useForColor'] = True
        if not 'useForGray' in f_keys:
            features[i]['fparams']['useForGray'] = True

        if ((features[i]['fparams']['useForColor'] and is_color_image) or
            (features[i]['fparams']['useForGray'] and  not is_color_image)):
            keep_features.append(features[i])

    features = keep_features

    for i in range(0,len(features)):
        if features[i]['name'] == 'get_fhog':
            if not 'nOrients' in features[i]["fparams"].keys():
                features[i]["fparams"]["nOrients"] = 9
            features[i]["fparams"]["nDim"] = np.array([3*features[i]["fparams"]["nOrients"] + 5 - 1])
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False

        elif features[i]['name'] == 'get_table_feature':
            cur_path = os.path.dirname(os.path.abspath(__file__))
            load_path = cur_path + '/lookup_tables' + features[i]["fparams"]["tablename"]
            table = scipy.io.loadmat(load_path)
            features[i]["fparams"]["nDim"] = table[features[i]["fparams"]["tablename"]].shape[1]
            features[i]["is_cell"] = False
            features[i]["ic_cnn"] = False

        elif features[i]['name'] == 'get_colorspace':
            features[i]["fparams"]["nDim"] = 1
            features[i]["is_cell"] = False
            features[i]["is_cnn"] = False

        elif features[i]['name'] == 'get_cnn_layers' or features[i]['name'] == 'get_OFcnn_layers':
            features[i]["fparams"]["output_layer"].sort()
            if not 'input_size_mode' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_mode"] = 'adaptive'
            if not 'input_size_scale' in features[i]["fparams"].keys():
                features[i]["fparams"]["input_size_scale"] = 1
            if not 'downsample_factor' in features[i]["fparams"].keys():
                features[i]["fparams"]["downsample_factor"] = np.ones((1, len(features[i]["fparams"]["output_layer"])))

            features[i]["fparams"]["nDim"] = np.array([96, 512]) #[64 512] net["info"]["dataSize"][layer_dim_ind, 2]

            features[i]["fparams"]["cell_size"] = np.array([4, 16]) #stride_tmp*downsample_factor

            features[i]["is_cell"] = True
            features[i]["is_cnn"] = True
        else:
            raise ValueError("Unknown feature type")

        if not 'cell_size' in features[i]["fparams"].keys():
            features[i]["fparams"]["cell_size"] = 1
        if not 'penalty' in features[i]["fparams"].keys():
            if len(features[i]["fparams"]["nDim"]) == 1:
                features[i]["fparams"]["penalty"] = 0
            else:
                features[i]["fparams"]["penalty"] = np.zeros((2, 1))
        features[i]["fparams"]["min_cell_size"] = np.min(features[i]["fparams"]["cell_size"])

    cnn_feature_ind = -1
    for i in range(0,len(features)):
        if features[i]["is_cnn"]:
            cnn_feature_ind = i  #last cnn feature

    if cnn_feature_ind >= 0 :
        # scale = features[cnn_feature_ind]["fparams"]["input_size_scale"]
        new_img_sample_sz = np.array(img_sample_sz, dtype=np.int32)

        if size_mode != "same" and features[cnn_feature_ind]["fparams"]["input_size_mode"] == "adaptive":
            orig_sz = np.ceil(new_img_sample_sz/16)

            if size_mode == "exact":
                desired_sz = orig_sz + 1
            elif size_mode == "odd_cells":
                desired_sz = orig_sz + 1 + orig_sz%2
            new_img_sample_sz = desired_sz*16

        if (features[cnn_feature_ind]["fparams"]["input_size_mode"] == "adaptive"):
            features[cnn_feature_ind]["img_sample_sz"] = np.round(new_img_sample_sz) # == feature_info.img_support_sz
        else:
            features[cnn_feature_ind]["img_sample_sz"] = np.array(img_sample_sz) #net["meta"].normalization.imageSize[0:2]

    for i in range(0, len(features)):
        if (not features[i]["is_cell"]):
            features[i]["img_sample_sz"] = features[cnn_feature_ind]["img_sample_sz"]
            features[i]["data_sz"] = np.ceil(features[i]["img_sample_sz"]/features[i]["fparams"]["cell_size"])
        else:
            features[i]["data_sz"] = np.ceil(features[i]["img_sample_sz"]/features[i]["fparams"]["cell_size"][:, None])
    return(features, gparams)
