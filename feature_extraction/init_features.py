import numpy as np
import scipy.io
import os
#from load_cnn import *
#from read_cnn import *
# from set_cnn_input_size import *

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
    # num_features = len(features)
    # feature_info = {}
    # feature_info['min_cell_size'] = np.zeros((num_features, 1))

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

            #net = load_cnn(features[i]["fparams"], img_sample_sz)
            # net["info"].blah_blah
            # layer_dim_ind = features[i]["fparams"]["output_layer"] - 1
            features[i]["fparams"]["nDim"] = np.array([96, 512]) #[64 512] net["info"]["dataSize"][layer_dim_ind, 2]

            # net_info_stride = []
            # if 'receptiveFieldStride' in net["info"].keys():
            #     shape = net["info"]["receptiveFieldStride"].shape
            #     net_info_stride = np.ones((shape[0]+1, shape[1]))
            #     net_info_stride[1:shape[0]+1,:] = net["info"]["receptiveFieldStride"]
            # else:
            #     net_info_stride = np.array([[1],[1]])

            # stride_tmp = []
            # # for layer in features[i]["fparams"]["output_layer"]:
            # #     stride_tmp.append(net_info_stride[0][layer + 1])
            
            # # stride_tmp = np.array(stride_tmp)
            # # stride_tmp = stride_tmp[np.newaxis]
            # # net_stride = net_stride
            # stride_tmp = net_info_stride[layer_dim_ind,0]
            # downsample_factor = features[i]["fparams"]["downsample_factor"][np.newaxis]
            # # downsample_factor = downsample_factor.T
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
        # feature_info["min_cell_size"][i] = np.min(features[i]["fparams"]["cell_size"])

    # features = [x for _, x in sorted(zip(feature_info["min_cell_size"], features))]
    # feature_info["min_cell_size"].sort()

    # feature_info["dim"] = []
    # feature_info["penalty"] = []

    # for i in range(0,len(features)):
    #     feature_info["dim"].append(features[i]["fparams"]["nDim"])
    #     feature_info["penalty"].append(features[i]["fparams"]["penalty"])
    # feature_info["dim"] = np.array(feature_info["dim"])
    # feature_info["penalty"] = feature_info["penalty"]

    cnn_feature_ind = -1
    for i in range(0,len(features)):
        if features[i]["is_cnn"]:
            cnn_feature_ind = i  #last cnn feature

    if cnn_feature_ind >= 0 :
        # scale = features[cnn_feature_ind]["fparams"]["input_size_scale"]
        new_img_sample_sz = np.array(img_sample_sz, dtype=np.int32)

        # net_info = net["info"]
        if size_mode != "same" and features[cnn_feature_ind]["fparams"]["input_size_mode"] == "adaptive":
            # orig_sz = net["info"]["dataSize"][-1,0:2]/features[cnn_feature_ind]["fparams"]["downsample_factor"][-1]
            orig_sz = np.ceil(new_img_sample_sz/16)

            if size_mode == "exact":
                desired_sz = orig_sz + 1
            elif size_mode == "odd_cells":
                desired_sz = orig_sz + 1 + orig_sz%2

            # while desired_sz[0] > net_info["dataSize"][-1, 0]:
            #     new_img_sample_sz += [1, 0]
            #     net_info = read_cnn(net, np.concatenate((np.round(scale*new_img_sample_sz), np.array([3, 1])), axis=None))

            # while desired_sz[1] > net_info["dataSize"][-1, 1]:
            #     new_img_sample_sz += [0, 1]
            #     net_info = read_cnn(net, np.concatenate((np.round(scale*new_img_sample_sz), np.array([3, 1])), axis=None))
            new_img_sample_sz = desired_sz*16

        # feature_info["img_sample_sz"] = np.round(new_img_sample_sz)

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
    #     scaled_sample_sz = np.round(scale*features[cnn_feature_ind]["img_input_sz"])

    #     net_info_stride = []
    #     if ('receptiveFieldStride' in net_info.keys()):
    #         net_info_stride = net_info["receptiveFieldStride"]
    #         net_info_stride = np.insert(net_info_stride, 0, [1, 1], axis=0)
    #     else:
    #         net_info_stride = np.array([1, 1])

    #     net_stride = net_info_stride[features[cnn_feature_ind]["fparams"]["output_layer"]]
    #     total_feat_sz = net_info["dataSize"][features[cnn_feature_ind]["fparams"]["output_layer"], 0:2]

    #     shrink_number = np.max(2*np.ceil((net_stride[-1]*total_feat_sz[-1] - scaled_sample_sz)/(2*net_stride[-1])), 0)
    #     deepest_layer_sz = total_feat_sz[-1] - shrink_number
    #     scaled_support_sz = net_stride[-1]*deepest_layer_sz
        
    #     cnn_output_sz = np.round(scaled_support_sz/net_stride)
    #     print cnn_output_sz
    #     features[cnn_feature_ind]["fparams"]["start_ind"] = np.floor((total_feat_sz - cnn_output_sz)/2.0) + 1
    #     features[cnn_feature_ind]["fparams"]["end_ind"] = features[cnn_feature_ind]["fparams"]["start_ind"] + cnn_output_sz - 1

    #     feature_info["img_support_sz"] = np.round(scaled_support_sz*feature_info["img_sample_sz"]/scaled_sample_sz)
    #     features[cnn_feature_ind]["fparams"]["net"] = set_cnn_input_size(net, feature_info["img_sample_sz"])

    #     # if gparams["use_gpu"]

    # else:
    #     max_cell_size = max(feature_info["min_cell_size"])
    #     if size_mode == 'same':
    #         feature_info["img_sample_size"] = np.round(img_sample_size)
    #     elif size_mode == 'exact':
    #         feature_info["img_sample_size"] = np.round(img_sample_size/max_cell_size) * max_cell_size
    #     elif size_mode == 'odd_cells':
    #         new_img_sample_sz = (1 + 2*np.round(img_sample_sz / (2*max_cell_size))) * max_cell_size
    #         feature_sz_choices = np.floor((new_img_sample_sz + np.array(range(0,max_cell_size)).reshape((max_cell_size, 1)).astype('float32'))/feature_info["min_cell_size"])
        
    #     print "Oops I didn't expect to get here. (NO cnn features??!!)"

    # feature_info["data_sz_block"] = []
    # for i in range(0, len(features)):
    #     if features[i]["is_cnn"]:
    #         features[i]["img_sample_sz"] = feature_info["img_sample_sz"]
    #         feature_info["data_sz_block"].append(np.floor(cnn_output_sz/np.tile(features[i]["fparams"]["downsample_factor"], (2,1)).T))
    #     else:
    #         features[i]["img_sample_sz"] = feature_info["img_support_sz"]
    #         features[i]["img_input_sz"] = features[i]["img_sample_sz"]
    #         feature_info["data_sz_block"].append(np.floor(features[i]["img_sample_sz"]/features[i]["fparams"]["cell_size"]))

    # # feature_info["data_sz"] = [item for sublist in feature_info["data_sz_block"] for item in sublist] # np.array?
    # feature_info["data_sz"] = feature_info["data_sz_block"]

    return(features, gparams)
