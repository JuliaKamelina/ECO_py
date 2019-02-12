import numpy as np

def get_reg_filter(sz, target_sz, params, reg_window_edge):
    if np.all(reg_window_edge == []):
        reg_window_edge = params["reg_window_edge"]

    if (params["use_reg_window"]):
        reg_window_power = params["reg_window_power"]

        reg_scale = 0.5*target_sz

        wrg = np.arange(-(sz[0]-1)/2, (sz[0]-1)/2)
        wcg = np.arange(-(sz[1]-1)/2, (sz[1]-1)/2)
        wrs, wcs = np.meshgrid(wrg, wcg)

        reg_window = (reg_window_edge - params["reg_window_min"]) * \
                     (np.abs(wrs/reg_scale[0])**reg_window_power + np.abs(wcs/reg_scale[1])**reg_window_power) + \
                     params["reg_window_min"]
        
        reg_window_dft = np.fft.fft(np.fft.fft(reg_window, axis=1), axis=0).astype(np.complex64) / np.prod(sz)
        
