import os
from pprint import pprint

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K


def calc_bandwidth(lambd, sigma):
    r = np.pi*sigma/lambd
    c = np.sqrt(np.log(2)/2)
    return np.log2((r + c)/(r - c))

def calc_sigma(lambd, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return lambd * c / np.pi  * (p + 1) / (p - 1)

def calc_lambda(sigma, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)

def plot_gabor_filters(params, image=None, fontsize=20, space=0.15, verbose=0):

    # ksize is the size of the Gabor kernel. If ksize = (a, b), we then have a Gabor kernel of size a x b pixels. As with many other convolution kernels, ksize is preferably odd and the kernel is a square (just for the sake of uniformity).
    # sigma is the standard deviation of the Gaussian function used in the Gabor filter.
    # theta is the orientation of the normal to the parallel stripes of the Gabor function.
    # lambda is the wavelength of the sinusoidal factor in the above equation.
    # gamma is the spatial aspect ratio.
    # psi is the phase offset.
    # ktype indicates the type and range of values that each pixel in the Gabor kernel can hold.

    if 'ksize' not in params:
        params['ksize'] = (31, 31)
    # sizes = [(31, 31)] #[(25, 25)]  # [(5, 5), (15, 15), (25, 25)]
    ksize = params['ksize']
    n_sizes = 1

    if 'thetas' not in params:  # Orientations
        # thetas = np.linspace(0, 2*np.pi, 8, endpoint=False)  # [0, np.pi/4, np.pi/2, np.pi*3/4]
        params['thetas'] = np.linspace(0, np.pi, 8, endpoint=False)
    thetas = params['thetas']
    n_thetas = len(params['thetas'])

    if 'psis' not in params:  # Phases
        # psis = [0, np.pi/2, np.pi, 3*np.pi/2]
        params['psis'] = np.linspace(0, 2*np.pi, 4, endpoint=False)
    psis = params['psis']
    n_psis = len(params['psis'])  # 1, 2, 4

    if 'gammas' not in params:  # Aspects
        # gamma = 0.5
        params['gammas'] = np.linspace(1, 0, 2, endpoint=False)
    gammas = params['gammas']
    n_gammas = len(params['gammas'])  # 2

    if 'sigmas' not in params:  # Sizes
        # sigmas = [2, 4]
        params['sigmas'] = [2, 4, 8, 16]
    sigmas = params['sigmas']
    n_sigmas = len(params['sigmas'])

    if 'lambdas' not in params:  # Wavelengths
        # lambdas = [3, 4, 5, 6, 7, 8]  # 3 <= lambd <= W/2
        params['lambdas'] = []
    lambdas = params['lambdas']
    n_lambdas = len(params['lambdas'])

    if 'bs' not in params:  # Bandwidths
        # bandwidths = np.linspace(0.4, 2.6, num=3)  # ~1.5 <= bw <= ~3
        params['bs'] = np.linspace(1, 1.8, num=3)
    bs = params['bs']
    n_bs = len(params['bs'])
    
    if verbose > 1:
        pprint(params)

    if image is not None:

        if isinstance(image, str):  # Assume file path is passed
            image = plt.imread(image)
            if image.shape[-1] == 3:
                # TODO: Generalise for channels first/RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = K.expand_dims(image, 0)  # For multiple images
                image = K.expand_dims(image, -1)  # For channels

        if isinstance(image, np.ndarray):
            image = image.astype('float32')

        if verbose:
            if verbose > 1:
                print(f"Image type: {type(image)}")
            print(f"Image shape: {image.shape}; scaling: [{np.amin(image)}, {np.amax(image)}]")

    # ncols = len(thetas)
    # nrows = int(np.ceil(2*len(sizes)*len(sigmas)*len(thetas)*len(lambdas)*len(psis)/ncols))
    ncols = n_thetas
    # nrows = int(np.ceil(n_sigmas * n_bs * n_gammas * n_psis * n_thetas / ncols))  #  * n_lambdas * n_sizes
    nrows = n_sigmas * n_bs * n_gammas * n_psis
    if verbose:
        print(f"Total Gabor filters: {ncols*nrows} "
              f"({n_sigmas} sigmas X {n_bs} bs X {n_gammas} gammas X {n_psis} psis X {n_thetas} thetas)")
    if image is not None:
        nrows *= 2
    
    if verbose > 1:
        print(nrows, ncols)

    width = 24

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='row', sharey='row', 
                             figsize=(width, width*nrows/ncols))

    i = 0
    for sg, sigma in enumerate(sigmas):
        for bw in bs:
        # for lm, lambd in enumerate(lambdas):
        #     sigma = calc_sigma(lambd, bw)
            lambd = calc_lambda(sigma, bw)
            if verbose > 1:
                print(f"sigma: {float(sigma):<3}, b: {bw}, lambda: {calc_lambda(sigma, bw):<05.3}")
            # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='row', sharey='row', figsize=(width, width*nrows/ncols))

            for gm, gamma in enumerate(gammas):
                for ps, psi in enumerate(psis):
                    for th, theta in enumerate(thetas):

                        p = {'ksize': ksize, 'sigma': sigma, 'theta': theta, 
                             'lambd': lambd, 'gamma': gamma, 'psi': psi}
                        gf = cv2.getGaborKernel(**p, ktype=cv2.CV_32F)

                        row, col = (i // ncols), i % ncols
                        if image is not None:
                            row *= 2

                        axes[row, col].imshow(gf, cmap='gray', vmin=-1, vmax=1)
                        axes[row, col].set_xticks([])
                        axes[row, col].set_yticks([])

                        if i // ncols == 0:  # On the first row
                            # simplify(th*np.pi/n_thetas)
                            if th == 0:
                                title = "$\\theta = 0$"
                            else:
                                title = f"$\\theta = \\frac{{{th}}}{{{n_thetas}}}\\pi$"
                            axes[row, col].set_title(title, fontsize=fontsize)

                        if i % ncols == 0:  # At the first column
                            # axes[row, col].set_ylabel(r"$\psi = {:.3}\pi, \gamma = {}$".format(psi/np.pi, gamma))  #lambd, sigma))
                            if ps == 0:
                                ylabel = (f"$\psi = 0, \enspace\gamma = {gamma}$\n"
                                          f"$b = {bw:.1f}, \lambda = {lambd:.1f}, \sigma = {float(sigma):.0f}$")
                            else:
                                # {:#.2g}  2 s.f. with trailing zeros
                                ylabel = (f"$\psi = \\frac{{{ps}}}{{{n_psis}}}\pi, \enspace\gamma = {gamma}$\n" 
                                          f"$b = {bw:.1f}, \lambda = {lambd:.1f}, \sigma = {float(sigma):.0f}$")

                            axes[row, col].set_ylabel(ylabel, fontsize=fontsize)

                        if image is not None:
                            gf = K.expand_dims(gf, -1)
                            gf = K.expand_dims(gf, -1)
                            # https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
                            # K.conv2d(image.img_to_array(img), gf)
                            fimg = K.conv2d(image, gf, padding='same')
                            fimg = fimg.numpy().squeeze()

                            # fimg = np.sign(fimg) * np.log(np.abs(fimg))  # Logarithmically compress convolved values
                            # fimg /= np.amax(np.abs(fimg))  # Scale to [-1, 1]
                            axes[row+1, col].imshow(fimg, cmap='gray') #, vmin=-img_scale, vmax=img_scale)
                            # axes[row+1, col].imshow(fimg[0,:,:,0].eval(), cmap='gray')
                            axes[row+1, col].set_xticks([])
                            axes[row+1, col].set_yticks([])

                            # if i % ncols == 0:
                            #     axes[row+1, col].set_ylabel(r"$\lambda = {:.2f}, \sigma = {:.0f}, b = {:.1f}$".format(lambd, float(sigma), bw), fontsize=fontsize)
                            
                        # if col + 1 == ncols:
                        #     ax_r = axes[row, col].twinx()
                        #     ax_r.set_ylabel(r"$\lambda = {:.2f}, \sigma = {:.0f}, b = {:.1f}$".format(lambd, float(sigma), bw), fontsize=fontsize)

                        i += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=space, hspace=space)
    # plt.savefig(os.path.join(fig_dir, f"gabor_kernels.pdf"), bbox_inches="tight")  # , additional_artists=[lgd])
    return fig, axes

def plot_dog_filters():

    # print(f"C: {channel}; sigma_c: {float(sigma):.1f}; r_sigma: {sig_ratio:.3f}; [{np.amin(dog):.5f}, {np.amax(dog):.5f}], Sum: {np.sum(dog):.5}", end='')
    pass