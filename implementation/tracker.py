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
from train import train_joint, train_filter
from optimize_scores import *

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
    feature_sz = np.array(feature_sz)

    filter_sz = feature_sz + (feature_sz + 1) % 2

    k_max = np.argmax(filter_sz)
    output_sz = filter_sz[k_max]  # The size of the label function DFT == maximum filter size

    block_inds = list(range(0, num_feature_blocks))
    block_inds.remove(k_max)

    #  How much each feature block has to be padded to the obtain output_sz
    pad_sz = []
    h = filter_sz.shape[0]
    for i in range(0, h):
        pad_sz.append((output_sz - filter_sz[i])/2)
    pad_sz = np.array(pad_sz)

    #  Compute the Fourier series indices and their transposes
    kx = [np.arange(-1*int(np.ceil(sz[0] - 1)/2.0), 1) for sz in filter_sz]
    ky = [np.arange(-1*int(np.ceil(sz[0] - 1)/2.0), int(np.ceil(sz[0] - 1)/2.0) + 1) for sz in filter_sz]

    #Gaussian label function
    sig_y = np.sqrt(np.prod(np.floor(base_target_sz))) * params["output_sigma_factor"] * (output_sz / img_support_sz)  # Gaussian label
    yf_y = [np.sqrt(2*math.pi)*sig_y[0]/output_sz[0]*np.exp(-2*(math.pi*sig_y[0]*y/output_sz[0])**2) for y in ky]
    yf_x = [np.sqrt(2*math.pi)*sig_y[1]/output_sz[1]*np.exp(-2*(math.pi*sig_y[1]*x/output_sz[1])**2) for x in kx]
    yf = [y.reshape(-1, 1)*x for y, x in zip(yf_y, yf_x)]
    
    cos_window = []
    for i in range(0, len(feature_sz)):
        cos_y = scipy.signal.hann(int(feature_sz[i][0]+2))
        cos_x = scipy.signal.hann(int(feature_sz[i][1]+2))
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

    reg_filter = np.array([get_reg_filter(img_support_sz, base_target_sz, params, reg_win_edge)
                           for reg_win_edge in reg_window_edge])
    # for i in range(0, len(reg_window_edge)):
    #     reg_filter.append(get_reg_filter(img_support_sz, base_target_sz, params)
    # reg_filter = np.array(reg_filter)

    reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                  for reg_filter in reg_filter]

    if params["use_scale_filter"]:
        print("Ops")
        raise NotImplementedError
    else:
        nScales = params["number_of_scales"]
        scale_step = params["scale_step"]
        scale_exp = np.arange(-np.floor((nScales-1)/2), np.ceil((nScales-1)/2)+1)
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
    # proj_matrix = 

    latest_ind = []
    frames_since_last_train = np.inf
    num_training_samples = 0

    minimum_sample_weight = params["learning_rate"] * (1 - params["learning_rate"])**(2*params["nSamples"])
    res_norms = []
    residuals_pcg = []

    tracker_time = 0
    while True:
        if seq["frame"] >= 0:
            if seq["frame"] >= seq["num_frames"]:
                im = []
                break
            else:
                im = cv.imread(seq["image_files"][seq["frame"]])
                if is_color_image:
                    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        # else:
        #     seq["frame"] = 0
        tic = time.clock()

        if seq["frame"] == 0:  # INIT AND UPDATE TRACKER
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

            if params["update_projection_matrix"]:
                # insert new sample
                for i in range(0, num_feature_blocks):
                    samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

            sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]

            # init CG params
            CG_state = None
            if params["update_projection_matrix"]:
                init_CG_opts['maxit'] = np.ceil(params["init_CG_iter"] / params["init_GN_iter"])
                hf = [[[]] * num_feature_blocks for _ in range(2)]
                feature_dim_sum = float(np.sum(feature_dim))
                proj_energy = [2 * np.sum(np.abs(yf_.flatten())**2) / feature_dim_sum * np.ones_like(P)
                                for P, yf_ in zip(proj_matrix, yf)]
            else:
                CG_opts['maxit'] = params["init_CG_iter"]
                hf = [[[]] * num_feature_blocks]

            # init filter
            for i in range(0, num_feature_blocks):
                hf[0][i] = np.zeros((int(filter_sz[i][0]), int((filter_sz[i][1]+1)/2), int(sample_dim[i]), 1), dtype=np.complex64)
            if params['update_projection_matrix']:
                # init gauss-newton optimiztion of filter and proj matrix
                hf, proj_matrix = train_joint(hf, proj_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts)
                xlf_proj = project_sample(xlf, proj_matrix, params["use_gpu"]) # reproject
                for i in range(0, num_feature_blocks):
                    samplesf[i][:, :, :, 0:1] = xlf_proj[i]  # insert new sample

                if params['distance_matrix_update_type'] == 'exact':
                    # find the norm of reproj sample
                    new_train_sample_norm = 0
                    for i in range(0, num_feature_blocks):
                        new_train_sample_norm += 2 * np.real(np.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                    gram_matrix[0, 0] = new_train_sample_norm
            hf_full = full_fourier_coeff(hf, params['use_gpu'])

            if params['use_scale_filter'] and nScales > 0:
                print("SCALE FILTER UPDATE")
                raise NotImplementedError
        else:   # TARGET LOCALIZATION
            old_pos = np.zeros((2))
            for _ in range(0, params['refinement_iterations']):
                if not np.allclose(old_pos, pos):
                    old_pos = pos
                    sample_pos = np.round(pos)
                    sample_scale = currentScaleFactor*scaleFactors
                    xt = [x for i in range(0, len(features))
                          for x in features[i]["feature"](im, features[i]["fparams"], global_fparams, sample_pos, features[i]['img_sample_sz'], currentScaleFactor)] # extract features
                    # if params['use_gpu']
                    xt_proj = project_sample(xt, proj_matrix, params['use_gpu'])  # project sample
                    xt_proj = [fmap * cos for fmap, cos in zip(xt_proj, cos_window)]  # do windowing
                    xtf_proj = [cfft2(x, params['use_gpu']) for x in xt_proj]  # fouries series
                    xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs)  # interpolate features

                    # compute convolution for each feature block in the fourier domain, then sum over blocks
                    scores_fs_feat = [[]]*num_feature_blocks
                    scores_fs_feat[k_max] = np.sum(hf_full[k_max]*xtf_proj[k_max], 2)
                    scores_fs = scores_fs_feat[k_max]

                    for ind in block_inds:
                        scores_fs_feat[ind] = np.sum(hf_full[ind]*xtf_proj[ind], 2)
                        scores_fs[int(pad_sz[ind][0]):int(output_sz[0]-pad_sz[ind][0]),
                                  int(pad_sz[ind][1]):int(output_sz[0]-pad_sz[ind][1])] += scores_fs_feat[ind]

                    # OPTIMIZE SCORE FUNCTION with Newnot's method.
                    trans_row, trans_col, scale_idx = optimize_scores(scores_fs, params["newton_iterations"], params['use_gpu'])

                    # compute the translation vector in pixel-coordinates and round to the cloest integer pixel
                    translation_vec = np.array([trans_row, trans_col])*(img_support_sz/output_sz)*currentScaleFactor*scaleFactors[scale_idx]
                    scale_change_factor = scaleFactors[scale_idx]

                    # update_position
                    pos = sample_pos + translation_vec

                    if params['clamp_position']:
                        pos = np.maximum(np.array(0, 0), np.minimum(np.array(im.shape[:2]), pos))

                    # do scale tracking with scale filter
                    if nScales > 0 and params['use_scale_filter']:
                        # scale_filter_track
                        raise(NotImplementedError)

                    # update scale
                    currentScaleFactor *= scale_change_factor

                    # adjust to make sure we are not to large or to small
                    if currentScaleFactor < min_scale_factor:
                        currentScaleFactor = min_scale_factor
                    elif currentScaleFactor > max_scale_factor:
                        currentScaleFactor = max_scale_factor

            # MODEL UPDATE STEP
            if params['learning_rate'] > 0:
                # use sample that was used for detection
                sample_scale = sample_scale[scale_idx]
                xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_proj]

                # shift sample target is centred
                shift_samp = 2*np.pi*(pos - sample_pos)/(sample_scale*img_support_sz)
                xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky, params['use_gpu'])

            # update the samplesf to include the new sample. The distance matrix, kernel matrix and prior weight are also updated
            merged_sample, new_sample, merged_sample_id, new_sample_id = update_sample_space_model(samplesf, xlf_proj, num_training_samples, distance_matrix, gram_matrix, prior_weights, params)
            if num_training_samples < params['nSamples']:
                num_training_samples += 1

            if params['learning_rate'] > 0:
                for i in range(0, num_feature_blocks):
                    if merged_sample_id >= 0:
                        samplesf[i][:,:,:,merged_sample_id:merged_sample_id+1] = merged_sample[i]
                    if new_sample_id >= 0:
                        samplesf[i][:,:,:,new_sample_id:new_sample_id+1] = new_sample[i]

            # train filter
            if seq['frame'] < params['skip_after_frame'] or frames_since_last_train >= params['train_gap']:
                new_sample_energy = [np.real(xlf * np.conj(xlf)) for xlf in xlf_proj]
                CG_opts['maxit'] = params['CG_iter']
                sample_energy = [(1 - params['learning_rate'])*se + params['learning_rate']*nse
                                 for se, nse in zip(sample_energy, new_sample_energy)]

                # do CG opt for filter
                hf, CG_state = train_filter(hf, samplesf, yf, reg_filter, prior_weights, sample_energy, reg_energy, params, CG_opts, CG_state)
                hf_full = full_fourier_coeff(hf)
                frames_since_last_train = 0
            else:
                frames_since_last_train += 1
            if params['use_scale_filter']:
                #scale_filter_update
                raise(NotImplementedError)

            # update target size
            target_sz = base_target_sz*currentScaleFactor
            tracker_time += time.clock() - tic
        bbox = (int(pos[1] - target_sz[1]/2),  # x_min
                int(pos[1] + target_sz[1]/2),  # x_max
                int(pos[0] - target_sz[0]/2),  # y_min
                int(pos[0] + target_sz[0]/2))  # y_max
        print(bbox)

        # VISUALIZATION
        # frame = im
        # if not is_color_image:
        #     frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        # frame = cv.rectangle(frame,
        #                       (int(bbox[0]), int(bbox[2])),
        #                       (int(bbox[1]), int(bbox[3])),
        #                       (0, 255, 255),
        #                       1)
        # frame = cv.putText(frame, str(seq["frame"]), (5, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # cv.imshow('',frame)
        seq["frame"] += 1
