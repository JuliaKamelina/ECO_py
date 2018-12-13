def init_feature_params(features, feature_info):
    num_features = len(features)
    feature_params = {}
    feature_params["compressed_dim_block"] = []
    for i in range(0, num_features):
        if not 'compressed_dim' in features[i]["fparams"].keys():
            features[i]["fparams"]["compressed_dim"] = features[i]["fparams"]["nDim"]
        feature_params["compressed_dim_block"].append([features[i]["fparams"]["compressed_dim"]])

    # feature_params["compressed_dim"] = [item for sublist in feature_params["compressed_dim_block"] for item in sublist]
    feature_params["compressed_dim"] = feature_params["compressed_dim_block"]
    return(feature_params)
