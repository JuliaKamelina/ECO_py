import numpy as np
import cv2 as cv
import math

def tracker(**params):
    #Get sequence info
    # TODO: when load image use from scipy.ndimage import imread
    seq, im = get_sequence_info(params["seq"])
    del params["seq"]
    if not im:
        seq["rect_position"] = []
        results = list()
        if (seq['format'] == 'otb'):
            results['type'] = 'rect'
            results['res'] = seq['rect_position']
        elif (seq['format'] == 'vot'):
            print("vot format")
            #seq.handle.quit(seq.handle)
        else:
            raise ValueError("Unknown sequence format")

        if ('time' in seq.keys()):
            results['fps'] = seq['num_frames'] / seq['time']
        else:
            results['fps'] = float('nan')
        #seq, results = get_sequence_results(seq)
        return

    #Init position
    pos = seq["init_pos"]
    target_sz = seq["init_sz"]
    params["init_sz"] = target_sz

    #Feature settings
    features = params["t_features"]

    #Set default parameters
    params = init_default_params(params)

    #Global feature params
    if "t_global" in params:
        global_fparams = params["t_global"]
    else:
        global_fparams = []
    global_fparams["use_gpu"] = params["use_gpu"]
    global_fparams["gpu_id"] = params["gpu_id"]

    #Correct max number of samples
    params["nSamples"] = min(params["nSamples"], seq["num_frames"])

    #Define data types
    if params["use_gpu"]:
        # TODO:
        params["data_type"] = pycuda.gpuarray.zeros(...)
    else:
        params["data_type"] = 0
    params["data_type_complex"] = complex(params["data_type"])
    global_fparams["data_type"] = params["data_type"]

    init_target_sz = target_sz

    # Check if colour image
    shape = im.shape
    if shape[2] == 3:
        is_color_image = True
    else:
        is_color_image = False

    # Check if mexResize available
    # TODO:

    # Calculate search area and initial scale factor
    search_area = np.prod(np.array(init_target_sz, float) * params["search_area_scale"])
    if search_area > params["max_image_sample_size"]:
        currentScaleFactor = math.sqrt(search_area / params["max_image_sample_size"])
    else:
        if  search_area < params["min_image_sample_size"]:
            currentScaleFactor = math.sqrt(search_area / params["min_image_sample_size"])
        else:
            currentScaleFactor = 1.0

    # target size at the initial scale
    base_target_sz = np.array(target_sz, float) / currentScaleFactor

    # window size, taking padding into account
    if params["search_area_shape"] == 'proportional':
        img_sample_sz = math.floor(base_target_sz * params["search_area_scale"])
    if params["search_area_shape"] == 'square':
        img_sample_sz = np.tile(math.sqtr(np.prod(base_target_sz*params["search_area_scale"])), 1, 2)[0]
    if params["search_area_shape"] == 'fix_padding':
        img_sample_sz = base_target_sz + math.sqrt(np.prod(base_target_sz*params["search_area_scale"]) + (base_target_sz[0] - base_target_sz[1])/4) - sum(base_target_sz)/2
    if params["search_area_shape"] == 'custom':
        img_sample_sz = np.array((base_target_sz[0]*2, base_target_sz[1]*2), float)

    # TODO:
    features, global_fparams, feature_info = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells')

    # Set feature info
    img_support_sz = feature_info["img_support_sz"]
    feature_sz = feature_info["data_sz"]
    feature_dim = feature_info["dim"]
    num_feature_blocks = len(feature_dim)

    # Get feature specific parameters
    # TODO:
    feature_params = init_feature_params(features, feature_info)
    feature_extract_info = get_feature_extract_info(features)

    if params["use_projection_matrix"]:
        sample_dim = feature_params["compressed_dim"]
    else:
        sample_dim = feature_dim

    # Size of the extracted feature maps
    h, w = feature_sz.shape
    feature_sz_cell = feature_sz.reshape(h//num_feature_blocks, num_feature_blocks, -1, 2)
                                         .swapaxes(1,2)
                                         .reshape(-1, num_feature_blocks, 2)
     #permute
