import numpy as np
import cv2 as cv
import math
import time
import sys

from scipy import signal

from init_features import *
from get_sequence_info import *
from get_feature_extract_info import *

from initialization.init_default_params import *
from initialization.get_interp_fourier import *
from initialization.get_reg_filter import *

from features import get_cnn_layers, get_fhog
from fourier_tools import *
from dim_reduction import *
from sample_space_model import update_sample_space_model

def tracker(params):
    #Get sequence info
    seq, im = get_sequence_info(params["seq"])
    del params["seq"]
    if len(im) == 0:
        seq["rect_position"] = []
        results = {}
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
    if "t_global" in params.keys():
        global_fparams = params["t_global"]
    else:
        global_fparams = {}
    global_fparams["use_gpu"] = params["use_gpu"]
    global_fparams["gpu_id"] = params["gpu_id"]

    #Correct max number of samples
    params["nSamples"] = min(params["nSamples"], seq["num_frames"])

    #Define data types
    if params["use_gpu"]:
        # TODO:
        params["data_type"] = "pycuda.gpuarray.zeros(...)"
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
    search_area = np.prod(init_target_sz * params["search_area_scale"])
    if search_area > params["max_image_sample_size"]:
        currentScaleFactor = math.sqrt(search_area / params["max_image_sample_size"])
    elif search_area < params["min_image_sample_size"]:
        currentScaleFactor = math.sqrt(search_area / params["min_image_sample_size"])
    else:
        currentScaleFactor = 1.0

    # target size at the initial scale
    base_target_sz = np.array(target_sz, float) / currentScaleFactor

    # window size, taking padding into account
    if params["search_area_shape"] == 'proportional':
        img_sample_sz = math.floor(base_target_sz * params["search_area_scale"])
    if params["search_area_shape"] == 'square':
        img_sample_sz = np.tile(math.sqrt(np.prod(base_target_sz*params["search_area_scale"])), (1, 2))[0]
    if params["search_area_shape"] == 'fix_padding':
        img_sample_sz = base_target_sz + math.sqrt(np.prod(base_target_sz*params["search_area_scale"]) + (base_target_sz[0] - base_target_sz[1])/4) - sum(base_target_sz)/2
    if params["search_area_shape"] == 'custom':
        img_sample_sz = np.array((base_target_sz[0]*2, base_target_sz[1]*2), float)

    features, global_fparams = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells')

    # Set feature info
    img_support_sz = features[0]["img_sample_sz"]
    feature_sz = []
    for i in range(0, len(features)):
        if len(features[i]["data_sz"].shape) > 1:
            for item in features[i]["data_sz"]:
                feature_sz.append(item)
        else:
            feature_sz.append(features[i]["data_sz"])
    feature_dim = [item for i in range(0, len(features)) for item in features[i]["fparams"]["nDim"]]
    num_feature_blocks = len(feature_dim)

    # Get feature specific parameters
    # feature_params = init_feature_params(features, feature_info)
    # feature_extract_info = get_feature_extract_info(features) TODO: CHECK

    if params["use_projection_matrix"]:
        sample_dim = [features[i]["fparams"]["compressed_dim"] for i in range(0, len(features))]
        sample_dim = np.concatenate((sample_dim[0], np.array(sample_dim[1]).reshape(1,)))
    else:
        sample_dim = feature_dim

    # Size of the extracted feature maps
    # TODO: Checks
    feature_sz_cell = []
    feature_sz_cell = np.array(feature_sz)

    filter_sz = []
    filter_sz = feature_sz_cell
    filter_sz = filter_sz + (filter_sz + 1) % 2

    k = np.argmax(filter_sz)
    output_sz = filter_sz[k]  # The size of the label function DFT

    # block_inds = np.arange(0, num_feature_blocks)
    # block_inds[k] = []

    #  How much each feature block has to be padded to the obtain output_sz
    pad_sz = []
    h = filter_sz.shape[0]
    for i in range(0, h):
        pad_sz.append((output_sz - filter_sz[i])/2)
    pad_sz = np.array(pad_sz)

    #  Compute the Fourier series indices and their transposes
    kx = []
    ky = []
    for i in range(0, len(filter_sz)):
        val = np.ceil(filter_sz[i][0] - 1)/2.0
        ky.append(np.arange(-1*int(val), int(val) + 1))
        kx.append(np.arange(-1*int(val), 1))
    kx = np.array(kx)
    ky = np.array(ky)

    #Gaussian label function
    sig_y = np.sqrt(np.prod(np.floor(base_target_sz))) * params["output_sigma_factor"] * (output_sz / img_support_sz)  # Gaussian label
    yf_y = []
    yf_x = []
    for i in range(0, len(kx)):
        yf_y.append(np.sqrt(2*math.pi)*sig_y[0]/output_sz[0]*np.exp(-2*(math.pi*sig_y[0]*ky[i]/output_sz[0])**2))
        yf_x.append(np.sqrt(2*math.pi)*sig_y[1]/output_sz[1]*np.exp(-2*(math.pi*sig_y[1]*kx[i]/output_sz[1])**2))
    yf_x = np.array(yf_x)
    yf_y = np.array(yf_y)
    yf = []
    for k in range(0, len(yf_y)):
        yf_k = []
        for i in range(0, len(yf_y[k])):
            yf_k.append(yf_y[k][i]*yf_x[k])
        yf.append(yf_k)
    yf = np.array(yf)
    
    cos_window = []
    for i in range(0, len(feature_sz_cell)):
        cos_y = scipy.signal.hann(int(feature_sz_cell[i][0]+2))
        cos_x = scipy.signal.hann(int(feature_sz_cell[i][1]+2))
        cos_x = cos_x.reshape(len(cos_x), 1)
        cos_window.append(cos_y*cos_x)
    cos_window = np.array(cos_window)
    for i in range(0, len(cos_window)):
        cos_window[i] = cos_window[i][1:-1,1:-1]
        cos_window[i] = cos_window[i].reshape(cos_window[i].shape[0], cos_window[i].shape[1], 1, 1)

    #Fourier for interpolation func
    interp1_fs = []
    interp2_fs = []
    for i in range(0, len(filter_sz)):
        interp1, interp2 = get_interp_fourier(filter_sz[i], params)
        interp1_fs.append(interp1.reshape(interp1.shape[0], 1, 1, 1))
        interp2_fs.append(interp2.reshape(interp2.shape[0], 1, 1, 1))
    interp1_fs = np.array(interp1_fs)
    interp2_fs = np.array(interp2_fs)

    reg_window_edge = np.array([])
    shape = 0
    for i in range(0, len(features)):
        shape += len(features[i]["fparams"]["nDim"])
    reg_window_edge = reg_window_edge.reshape((shape, 0))

    reg_filter = []
    for i in range(0, len(reg_window_edge)):
        reg_filter.append(get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge[i]))
    reg_filter = np.array(reg_filter)

    reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                    for reg_filter in reg_filter]

    if params["use_scale_filter"]:
        print("Ops")
    else:
        nScales = params["number_of_scales"]
        scale_step = params["scale_step"]
        scale_exp = np.arange(-np.floor((nScales-1)/2), np.ceil((nScales-1)/2))
        scaleFactors = scale_step**scale_exp
    
    if nScales > 0:
        # force reasonable scale changes
        min_scale_factor = scale_step ** np.ceil(np.log(np.max(5 / img_support_sz)) / np.log(scale_step))
        max_scale_factor = scale_step ** np.floor(np.log(np.min(im.shape[:2] / base_target_sz)) / np.log(scale_step))

    init_CG_opts = {
        "CG_use_FR": True,
        "tol": 1e-6,
        "CG_standard_alpha": True,
        "debug": params["debug"]
    }
    CG_opts = {
        "CG_use_FR": params["CG_use_FR"],
        "tol": 1e-6,
        "CG_standard_alpha": params["CG_standard_alpha"],
        "debug": params["debug"]
    }
    if params["CG_forgetting_rate"] == np.inf or params["learning_rate"] >= 1:
        CG_opts["init_forget_factor"] = 0
    else:
        CG_opts["init_forget_factor"] = (1-params["learning_rate"])**params["CG_forgetting_rate"]

    #init and alloc
    prior_weights = np.zeros((params["nSamples"],1), dtype=np.float32)
    sample_weights = prior_weights
    samplesf = [[]] * num_feature_blocks
    for i in range(num_feature_blocks):
        samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                      sample_dim[i], params["nSamples"]), dtype=np.complex64)
    
    scores_fs_feat = [[]] * num_feature_blocks

    distance_matrix = np.ones((params["nSamples"], params["nSamples"]), dtype=np.float32) * np.inf  # stores the square of the euclidean distance between each pair of samples
    gram_matrix = np.ones((params["nSamples"], params["nSamples"]), dtype=np.float32) * np.inf  # Kernel matrix, used to update distance matrix

    latest_ind = []
    frames_since_last_train = np.inf
    num_training_samples = 0

    minimum_sample_weight = params["learning_rate"] * (1 - params["learning_rate"])**(2*params["nSamples"])
    res_norms = []
    residuals_pcg = []

    while True:
        if seq["frame"] > 0:
            if seq["frame"] >= seq["num_frames"]:
                im = []
                break
            else:
                im = cv.imread(seq["image_files"][seq["frame"]])
            seq["frame"] += 1
        else:
            seq["frame"] = 0
        tic = time.clock()

        if seq["frame"] == 0:
            sample_pos = np.round(pos)
            sample_scale = currentScaleFactor
            xl = [x for i in range(0, len(features))
                    for x in features[i]["feature"](im, features[i]["fparams"], global_fparams, sample_pos, features[i]['img_sample_sz'], currentScaleFactor)]
            # print(xl)

            xlw = [x * y for x, y in zip(xl, cos_window)]      # do windowing of feature
            xlf = [cfft2(x) for x in xlw]                      # compute the fourier series
            xlf = interpolate_dft(xlf, interp1_fs, interp2_fs) # interpolate features
            xlf = compact_fourier_coeff(xlf)                   # new sample to add
            # shift sample
            shift_samp = 2 * np.pi * (pos - sample_pos) / (sample_scale * img_support_sz) # img_sample_sz
            xlf = shift_sample(xlf, shift_samp, kx, ky)
            proj_matrix = init_projection_matrix(xl, sample_dim, params['proj_init_method'])  # init projection matrix
            xlf_proj = project_sample(xlf, proj_matrix)  # project sample

            merged_sample, new_sample, merged_sample_id, new_sample_id = update_sample_space_model(samplesf, xlf_proj, num_training_samples, 
                                                                                                    distance_matrix, gram_matrix, prior_weights, params)
            num_training_samples += 1
