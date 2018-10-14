def init_default_params(**params):
    default_params = list()  # TODO: check default_params
    default_params["use_gpu"] = False
    default_params["gpu_id"] = []
    default_params["interpolation_method"] = 'none'
    default_params["interpolation_centering"] = False
    default_params["interpolation_windowing"] = False
    default_params["clamp_position"] = False
    default_params["update_projection_matrix"] = True
    default_params["proj_init_method"] = 'pca'
    default_params["use_detection_sample"] = True
    default_params["use_projection_matrix"] = True
    default_params["use_sample_merge"] = True
    default_params["CG_use_FR"] = True
    default_params["CG_standard_alpha"] = true

    def_param_names = default_params.keys()
    for param_name in def_param_names:
        if !(param in params.keys()):
            params[param_name] = default_params[param_name]

    if params["use_projection_matrix"] == False
        params["proj_init_method"] = 'none'
        params["update_projection_matrix"] = False

    params["visualization"] = params["isualization"] | params["debug"]
    return(params)
