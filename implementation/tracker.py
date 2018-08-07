def tracker(**params):
    #Get sequence info
    seq, im = get_sequence_info(params["seq"])
    del params["seq"]
    if not im:
        seq["rect_position"] = []
        seq, results = get_sequence_results(seq)
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
        params["data_type"] = pycuda.gpuarray.zeros(...)
    else:
        params["data_type"] = 0
    params["data_type_complex"] = complex(params["data_type"])
    global_fparams["data_type"] = params["data_type"]

    init_target_sz = target_sz
