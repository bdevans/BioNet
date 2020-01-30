import os
from pprint import pprint

import numpy as np
from scipy import signal
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D  # , Dense
from tensorflow.keras.initializers import Initializer
from tensorflow.python.framework import dtypes


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

def plot_gabor_filters(params, images=None, use_gpu=True, fontsize=20, space=0.15, verbose=0):

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

    if images is not None:

        if not isinstance(images, list):
            assert isinstance(images, str) or isinstance(images, np.ndarray)
            images = [images]
        
        n_images = len(images)

        for i, image in enumerate(images):
        if isinstance(image, str):  # Assume file path is passed
            image = plt.imread(image)
            if image.shape[-1] == 3:
                # TODO: Generalise for channels first/RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if isinstance(image, np.ndarray):
            image = image.astype('float32')

        if use_gpu:
            image = K.expand_dims(image, 0)  # For multiple images
            image = K.expand_dims(image, -1)  # For channels

        if verbose:
            if verbose > 1:
                print(f"Image type: {type(image)}")
            print(f"Image shape: {image.shape}; scaling: [{np.amin(image)}, {np.amax(image)}]")

            images[i] = image

    # ncols = len(thetas)
    # nrows = int(np.ceil(2*len(sizes)*len(sigmas)*len(thetas)*len(lambdas)*len(psis)/ncols))
    ncols = n_thetas
    # nrows = int(np.ceil(n_sigmas * n_bs * n_gammas * n_psis * n_thetas / ncols))  #  * n_lambdas * n_sizes
    nrows = n_sigmas * n_bs * n_gammas * n_psis
    wide_format = False
    if n_psis == 2:
        wide_format = True
        ncols *= n_psis
        nrows = int(nrows / n_psis)
    if verbose:
        print(f"Total Gabor filters: {ncols*nrows} "
              f"({n_sigmas} sigmas X {n_bs} bs X {n_gammas} gammas X {n_psis} psis X {n_thetas} thetas)")
        lambda_min = calc_lambda(np.amin(sigmas), np.amin(bs))
        lambda_max = calc_lambda(np.amax(sigmas), np.amax(bs))
        print(f"Wavelength range: [{lambda_min:#.3g}, {lambda_max:#.3g}] pixels")
    if images is not None:
        nrows *= (n_images + 1)
    
    if verbose > 1:
        print(nrows, ncols)

    width = 24

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='row', sharey='row', 
                             figsize=(width, width*nrows/ncols), squeeze=False)

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

                        if images is not None:
                            row *= (n_images + 1)

                        axes[row, col].imshow(gf, cmap='gray', vmin=-1, vmax=1)
                        axes[row, col].set_xticks([])
                        axes[row, col].set_yticks([])

                        if wide_format:
                            if i // ncols == 0:  # On the first row
                                if th == 0:
                                    theta_str = "\\theta = 0"
                                else:
                                    theta_str = f"\\theta = \\frac{{{th}}}{{{n_thetas}}}\\pi"
                                if ps == 0:
                                    psi_str = "\\psi = 0"
                                else:
                                    psi_str = f"\psi = \\frac{{{ps}}}{{{n_psis}}}\pi"
                                title = f"${theta_str}, \enspace{psi_str}$"
                                axes[row, col].set_title(title, fontsize=fontsize)

                            if i % ncols == 0:  # At the first column
                                ylabel = (f"$\gamma = {gamma}, \enspace b = {bw:.1f}$\n"
                                          f"$\lambda = {lambd:.1f}, \enspace \sigma = {float(sigma):.0f}$")
                                axes[row, col].set_ylabel(ylabel, fontsize=fontsize)
                        else:
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

                        if images is not None:

                            if use_gpu:
                                gf = K.expand_dims(gf, -1)
                                gf = K.expand_dims(gf, -1)

                            for ind, image in enumerate(images):
                                if use_gpu:
                                # https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
                                # K.conv2d(image.img_to_array(img), gf)
                                fimg = K.conv2d(image, gf, padding='same')
                                fimg = fimg.numpy().squeeze()
                            else:
                                fimg = signal.convolve2d(image, gf, mode='same')
                                # fimg = cv2.filter2D(image, -1, gf)  # Alternative

                            # fimg = np.sign(fimg) * np.log(np.abs(fimg))  # Logarithmically compress convolved values
                            # fimg /= np.amax(np.abs(fimg))  # Scale to [-1, 1]
                                axes[row+1+ind, col].imshow(fimg, cmap='gray') #, vmin=-img_scale, vmax=img_scale)
                            # axes[row+1, col].imshow(fimg[0,:,:,0].eval(), cmap='gray')
                                axes[row+1+ind, col].set_xticks([])
                                axes[row+1+ind, col].set_yticks([])

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


def get_gabor_tensor(ksize, bs, sigmas, thetas, gammas, psis, lambdas=None):
    """Create a tensor of Gabor filters for greyscale images.

    The sinusoidal wavelength $\lambda$ is constrained by the bandwidth $b$
    and Gaussian spatial scale $\sigma$.
    """

    # n_kernels = len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    n_kernels = len(bs) * len(sigmas) * len(thetas) * len(gammas) * len(psis)
    gabors = []
    for sigma in sigmas:
        for theta in thetas:
            # for lambd in lambdas:
            for b in bs:
                lambd = calc_lambda(sigma, b)
                for gamma in gammas:
                    for psi in psis:
                        params = {'ksize': ksize, 'sigma': sigma,
                                  'theta': theta, 'lambd': lambd,
                                  'gamma': gamma, 'psi': psi}
                        gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = K.expand_dims(gf, -1)
                        gabors.append(gf)
    assert len(gabors) == n_kernels
    print(f"Created {n_kernels} kernels.")
    return K.stack(gabors, axis=-1)  # (ksize[0], ksize[1], 1, n_kernels)


@tf.function
def convolve_tensor(x, kernel_tensor=None):
    '''
    conv2d
    input tensor: [batch, in_height, in_width, in_channels]
    kernel tensor: [filter_height, filter_width, in_channels, out_channels]
    '''
    # x = tf.image.rgb_to_grayscale(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Kernel tensor shape: {kernel_tensor.shape}")  # TODO: Should this be (127, 127, 3, n)?

    # NOTE: This function does not apply bias or pass through an activation function
    # TODO: Do I need to add bias and pass through an activation function before returning? Maybe...
    return K.conv2d(x, kernel_tensor, padding='same')


def substitute_layer(model, params, filter_type='gabor', replace_layer=1, colour_input='rgb', test=False):

    assert isinstance(replace_layer, int)
    assert 0 < replace_layer < len(model.layers)

    # Parse parameters
    if params is not None:
    assert 'bs' in params
    if 'sigmas' not in params:
        assert 'lambdas' in params

    #     params['sigmas'] = [utils.calc_sigma(lambd, b) for lambd in params['lambdas']
    #                         for b in params['bs']]
    
    # if 'sigmas' in params and 'lambdas' not in params:
    #     assert 'bs' in params
    #     params['lambdas'] = [utils.calc_lambda(sigma, b) for sigma in params['sigmas']
    #                          for b in params['bs']]

    print(f"{filter_type.capitalize()} filter parameters:")
    pprint(params)

    # Generate Gabor filters
    # tensor = get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)
    tensor = get_gabor_tensor(**params)

    for ind, layer in enumerate(model.layers):
        if ind == 0:  # Get input layer
            config = layer.get_config()
            print(f"Original (batch) input shape: {config['batch_input_shape']}")
            if colour_input == 'rgb':
                # inp = Input(shape=model.layers[0].input_shape[0][1:])
                config['shape'] = config['batch_input_shape'][1:]
            elif colour_input == 'rgba':
                print(f"Warning! colour_input: {colour_input} not yet implemented!")
                return
            elif colour_input == "grayscale":
                original_shape = config['batch_input_shape'][1:]
                config['shape'] = (*original_shape[:-1], 1)
            else:
                raise UserError(f"Unknown colour_input: {colour_input}")
            del config['batch_input_shape']
            inp = Input(**config)
            x = inp
        elif ind == replace_layer and params is not None:  # Replace convolutional layer
            print(f"Replacing layer {ind}: '{layer.name}' --> '{filter_type}_conv'...")
                n_kernels = len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
                                              * len(params['gammas']) * len(params['psis'])
                
                # When using this layer as the first layer in a model, provide the keyword argument 
                # input_shape (tuple of integers, does not include the batch axis), 
                # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

                # Input shape: (batch, rows, cols, channels)
                # Output shape: (batch, new_rows, new_cols, filters)
                x = Conv2D(n_kernels, params['ksize'], padding='same', activation='relu', use_bias=True,
                           kernel_initializer=GaborInitializer(params))(x)
                # x = Conv2D(n_kernels, params['ksize'], padding='same', activation='relu', use_bias=True)(x)

            else:
                assert isinstance(layer, tf.keras.layers.Conv2D)
                x = Lambda(convolve_tensor, arguments={'kernel_tensor': tensor},
                        name=f"{filter_type}_conv")(x)
        elif ind == replace_layer + 1 and params is not None:  # Replace next layer
            # Check input_shape matches output_shape?
            # x = Conv2D(**layers[layer].get_config())(x)
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)
        else:
            x = layer(x)

    if test:
        # Freeze weights of kernels
        model = Model(inputs=inp, outputs=x, name=f"{filter_type}_{model.name}")
        model.layers[replace_layer].trainable = False
        return model

    return Model(inputs=inp, outputs=x, name=f"{filter_type}_{model.name}")


def substitute_output(model, n_classes=16):

    if model.get_layer(index=-1).output_shape[-1] == n_classes:
        print(f"Model already has the requested number of output classes ({n_classes})!")
        return model

    new_model = []
    n_layers = len(model.layers)

    for ind, layer in enumerate(model.layers):
        if ind == 0:
            params = layer.get_config()
            params['shape'] = params['batch_input_shape'][1:]
            del params['batch_input_shape']
            inp = Input(**params)
            x = inp
        elif 0 < ind < n_layers-1:
            x = layer(x)
        else:  # Final layer
            params = layer.get_config()
            params['units'] = n_classes
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': params})(x)

    return Model(inputs=inp, outputs=x, name=f"{model.name}_{n_classes}class")


# def add_top(model, n_fc_layers=3, n_classes=16):

#     new_model = []
#     for ind, layer in enumerate(model.layers):
#         new_model.append(layer)
    
#     for ind in range(n_fc_layers):
#         if ind < n_fc_layers-1:
#             name = f"fc{ind+1}"
#         else:
#             name = "predictions"
#         new_model.append(Dense())


class GaborInitializer(Initializer):
    """Gabor Initializer class."""

    def __init__(self, params, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)
        self.params = params
        # self.n_kernels = self.calc_n_kernels(params)
    
    def calc_n_kernels(params):
        return len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
                                 * len(params['gammas']) * len(params['psis'])

    def __call__(self, shape, dtype=None, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        # return get_gabor_tensor(ksize, bs, sigmas, thetas, gammas, psis, lambdas=None)
        return get_gabor_tensor(**self.params)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        return self.params
