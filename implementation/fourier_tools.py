import numpy as np

def fft2(x, use_gpu = False):
    if use_gpu:
        import cupy as cp

        print('use gpu')
        # xp = cp.get_array_module(x)
    else:
        xp = np
    return xp.fft.fft(xp.fft.fft(x, axis=1), axis=0).astype(xp.complex64)

def ifft2(x, use_gpu = False):
    if use_gpu:
        import cupy as cp

        print('use gpu')
        # xp = cp.get_array_module(x)
    else:
        xp = np
    return xp.fft.ifft(xp.fft.ifft(x, axis=1), axis=0).astype(xp.complex64)

def cfft2(x, use_gpu = False):
    in_shape = x.shape
    # if both dimensions are odd
    if use_gpu:
        import cupy as cp
        print('use gpu')
        # xp = cp.get_array_module(x)
    else:
        xp = np
    if in_shape[0] % 2 == 1 and in_shape[1] % 2 == 1:
        xf = xp.fft.fftshift(xp.fft.fftshift(fft2(x, use_gpu), 0), 1).astype(xp.complex64)
    else:
        out_shape = list(in_shape)
        out_shape[0] =  in_shape[0] + (in_shape[0] + 1) % 2
        out_shape[1] =  in_shape[1] + (in_shape[1] + 1) % 2
        out_shape = tuple(out_shape)
        xf = xp.zeros(out_shape, dtype=xp.complex64)
        xf[:in_shape[0], :in_shape[1]] = xp.fft.fftshift(xp.fft.fftshift(fft2(x, use_gpu), 0), 1).astype(xp.complex64)
        if out_shape[0] != in_shape[0]:
            xf[-1,:] = xp.conj(xf[0,::-1])
        if out_shape[1] != in_shape[1]:
            xf[:,-1] = xp.conj(xf[::-1,0])
    return xf

def interpolate_dft(xf, interp1_fs, interp2_fs):
    return [xf_ * interp1_fs_ * interp2_fs_
            for xf_, interp1_fs_, interp2_fs_ in zip(xf, interp1_fs, interp2_fs)]

def compact_fourier_coeff(xf):
    if isinstance(xf, list):
        return [x[:, :(x.shape[1]+1)//2, :] for x in xf]
    else:
        return xf[:, :(xf.shape[1]+1)//2, :]

def shift_sample(xf, shift, kx, ky, use_gpu=False):
    if use_gpu:
        print('use gpu')
        #xp = cp.get_array_module(xf[0])
    else:
        xp = np
    shift_exp_y = [xp.exp(1j * shift[0] * y).astype(xp.complex64) for y in ky]
    shift_exp_x = [xp.exp(1j * shift[1] * x).astype(xp.complex64) for x in kx]
    xf = [xf_ * sy_.reshape(-1, 1, 1, 1) * sx_.reshape((1, -1, 1, 1))
            for xf_, sy_, sx_ in zip(xf, shift_exp_y, shift_exp_x)]
    return xf

def symmetrize_filter(hf, use_gpu=False):
    if use_gpu:
        import cupy as cp
        print("GPU")
        xp = cp.get_array_module(hf[0])
    else:
        xp = np

    for i in range(len(hf)):
        dc_ind = int((hf[i].shape[0]+1) / 2)
        hf[i][dc_ind:, -1, :] = xp.conj(xp.flipud(hf[i][:dc_ind-1, -1, :]))
    return hf

def full_fourier_coeff(xf, use_gpu=False):
    """
        reconstruct full fourier series
    """

    if use_gpu:
        xp = cp.get_array_module(xf[0])
    else:
        xp = np
    xf = [xp.concatenate([xf_, xp.conj(xp.rot90(xf_[:, :-1,:], 2))], axis=1) for xf_ in xf]
    return xf
