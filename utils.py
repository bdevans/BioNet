"""
This module contains help functions and general utilities for building custom convolutional layers.
"""

import os
import json
from pprint import pprint

import numpy as np
# from scipy import signal
import cv2
# from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D  # , Dense
from tensorflow.keras.initializers import Initializer
from tensorflow.python.framework import dtypes
# from K.tensorflow_backend import set_session
# from K.tensorflow_backend import clear_session
# from K.tensorflow_backend import get_session


# # Reset Keras Session
# def reset_keras():
#     sess = get_session()
#     clear_session()
#     sess.close()
#     sess = get_session()

#     try:
#         del classifier # this is from global space - change this as you need
#     except:
#         pass

#     print(gc.collect()) # if it's done something you should see a number being outputted

#     # use the same config as you used to create the session
#     config = tensorflow.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 1
#     config.gpu_options.visible_device_list = "0"
#     set_session(tensorflow.Session(config=config))


def get_simulation_params(sim_set, model_name, models_dir='/work/models'):
    with open(os.path.join(models_dir, sim_set, model_name, 'simulation.json'), 'r') as fh:
        sim = json.load(fh)
    return sim


def load_images(path):

    image_set = {}
    for root, dirs, files in os.walk(path):
        if root == path:
            categories = sorted(dirs)
            image_set = {cat: [] for cat in categories}
        else:
            image_set[os.path.basename(root)] = sorted(files)

    n_cat_images = {cat: len(files) for (cat, files) in image_set.items()}
    n_images = sum(n_cat_images.values())
    image_dims = plt.imread(os.path.join(path, categories[0],
                            image_set[categories[0]][0])).shape

    print(image_dims)
    X = np.zeros((n_images, image_dims[0], image_dims[1], 1), dtype='float16')  # 'float32'
    y = np.zeros((n_images, len(categories)), dtype=int)

    tally = 0
    for c, (cat, files) in enumerate(tqdm(image_set.items(), desc=path)):
        for i, image in enumerate(files):
            cimg = cv2.imread(os.path.join(path, cat, image))  # cv2 opens in BGR
            X[i+tally] = np.expand_dims(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), axis=-1)
            # Alternative in one operation
            # X[i+tally] = np.expand_dims(cv2.imread(os.path.join(path, cat, image), 
            #                                        cv2.IMREAD_GRAYSCALE), axis=-1)
        y[tally:tally+len(files), c] = True
        tally += len(files)

    shuffle = np.random.permutation(y.shape[0])

    return image_set, X[shuffle], y[shuffle]


# def upscale_images(images, size, interpolation=None):
# """Images: (n_images, height, width, n_channels)"""
#     if interpolation is None:
#         x_train = x_train.repeat(7, axis=1).repeat(7, axis=2)
#         x_test = x_test.repeat(7, axis=1).repeat(7, axis=2)
#     else:
#         for image in images:
#             resized = cv2.resize(img, dim, interpolation=interpolation)

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
                        # params = {'ksize': ksize, 'sigma': sigma,
                        #           'theta': theta, 'lambd': lambd,
                        #           'gamma': gamma, 'psi': psi}
                        # gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = cv2.getGaborKernel(ksize, sigma, theta, 
                                                lambd, gamma, psi, 
                                                ktype=cv2.CV_32F)
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


def substitute_layer(model, params, filter_type='gabor', replace_layer=1, 
                     input_shape=None, colour_input='rgb', 
                     use_initializer=False, verbose=0):

    if replace_layer is None:
        # Attempt to find the first convolutional layer
        for ind, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"Found first convolutional layer {ind}: {layer.name}")
                replace_layer = ind
                break
    assert isinstance(replace_layer, int)
    assert 0 < replace_layer < len(model.layers)

    # Parse parameters
    if params is not None and filter_type.capitalize() == 'Gabor':
        assert 'bs' in params
        if 'sigmas' not in params:
            assert 'lambdas' in params

    #     params['sigmas'] = [utils.calc_sigma(lambd, b) for lambd in params['lambdas']
    #                         for b in params['bs']]
    
    # if 'sigmas' in params and 'lambdas' not in params:
    #     assert 'bs' in params
    #     params['lambdas'] = [utils.calc_lambda(sigma, b) for sigma in params['sigmas']
    #                          for b in params['bs']]

    if verbose:
        print(f"{filter_type.capitalize()} filter parameters:")
        pprint(params)

    for ind, layer in enumerate(model.layers):
        # print(ind, layer.name)
        if ind == 0:  # Get input layer
            config = layer.get_config()
            print(f"Replacing input: {config['batch_input_shape']} (batch) -->", end=' ')
            if colour_input == 'rgb':
                # inp = Input(shape=model.layers[0].input_shape[0][1:])
                if input_shape:
                    config['shape'] = input_shape + (3,)
                else:
                    config['shape'] = config['batch_input_shape'][1:]
            elif colour_input == 'rgba':
                print(f"Warning! colour_input: {colour_input} not yet implemented!")
                return
            elif colour_input == "grayscale":
                if input_shape:
                    config['shape'] = input_shape + (1,)
                else:
                    original_shape = config['batch_input_shape'][1:]
                    config['shape'] = (*original_shape[:-1], 1)
            else:
                raise UserError(f"Unknown colour_input: {colour_input}")
            del config['batch_input_shape']
            inp = Input(**config)
            # inp = layer
            x = inp
            print(f"{config['shape']}")
        elif ind == replace_layer and params is not None:  # Replace convolutional layer
            print(f"Replacing layer {ind}: '{layer.name}' --> '{filter_type.lower()}_conv'...")
            if use_initializer:
                if filter_type.lower() == 'gabor':
                    n_kernels = len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
                                                  * len(params['gammas']) * len(params['psis'])
                    kernel_initializer = GaborInitializer(params)
                elif filter_type.lower() == 'low-pass':
                    n_kernels = len(params['sigmas'])
                    kernel_initializer = LowPassInitializer(params)
                # When using this layer as the first layer in a model, provide the keyword argument 
                # input_shape (tuple of integers, does not include the batch axis), 
                # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

                # Input shape: (batch, rows, cols, channels)
                # Output shape: (batch, new_rows, new_cols, filters)
                x = Conv2D(n_kernels, params['ksize'], padding='same', 
                           activation='relu', use_bias=True,
                        #    activation=None, use_bias=False,
                           name=f"{filter_type.lower()}_conv",
                           kernel_initializer=kernel_initializer)(x)
                # x = Conv2D(n_kernels, params['ksize'], padding='same', activation='relu', use_bias=True)(x)

            else:  # Deprecated
                assert isinstance(layer, tf.keras.layers.Conv2D)
                # Generate Gabor filters
                # tensor = get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)
                tensor = get_gabor_tensor(**params)
                x = Lambda(convolve_tensor, arguments={'kernel_tensor': tensor},
                           name=f"{filter_type.lower()}_conv")(x)
        # elif ind == replace_layer + 1 and params is not None:  # Replace next layer
        #     # Check input_shape matches output_shape?
        #     # x = Conv2D(**layers[layer].get_config())(x)
        #     x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
        #                                      'config': layer.get_config()})(x)
        else:
            # x = layer(x)
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)
        # print(x.shape)

    # del model
    model = Model(inputs=inp, outputs=x, name=f"{filter_type}_{model.name}")
    if use_initializer:
        # Freeze weights of kernels
        # model = Model(inputs=inp, outputs=x, name=f"{filter_type}_{model.name}")
        model.layers[replace_layer].trainable = False
    return model


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
    """Gabor kernel initializer class."""

    # def __init__(self, params, dtype=dtypes.float32):
    #     self.dtype = dtypes.as_dtype(dtype)
    #     self.params = params
    #     # self.n_kernels = self.calc_n_kernels(params)
    
    def __init__(self, ksize, sigmas, bs, gammas, thetas, psis):
        # TODO: Deprecate ksize in (initialisation of) filter parameters
        if isinstance(ksize, (int, float)):
            self.ksize = (ksize, ksize)
        else:
            self.ksize = tuple(ksize)
        self.sigmas = sigmas
        self.bs = bs
        self.gammas = gammas
        self.thetas = thetas
        self.psis = psis
        self.n_kernels = len(sigmas) * len(bs) * len(gammas) * len(thetas) * len(psis)
    
    # def calc_n_kernels(params):
    #     return len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
    #                              * len(params['gammas']) * len(params['psis'])

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        # return get_gabor_tensor(ksize, bs, sigmas, thetas, gammas, psis, lambdas=None)
        # return get_gabor_tensor(**self.params)
        if shape is None:
            ksize = self.ksize
        else:
            ksize = tuple(shape[:2])
            assert self.n_kernels == shape[-1]
        return get_gabor_tensor(ksize, self.bs, self.sigmas, 
                                self.thetas, self.gammas, self.psis)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        # return self.params
        return {'ksize': self.ksize,
                'sigmas': self.sigmas,
                'bs': self.bs,
                'gammas': self.gammas,
                'thetas': self.thetas,
                'psis': self.psis
                }

    
# from preparation import low_pass_filter

class LowPassInitializer(Initializer):
    """Low Pass filter Initializer class."""

    def __init__(self, ksize, sigmas):  #params, dtype=dtypes.float32):
        # self.dtype = dtypes.as_dtype(dtype)
        # self.params = params
        # TODO: Deprecate ksize in (initialisation of) filter parameters
        if isinstance(ksize, (int, float)):
            self.ksize = (ksize, ksize)
        else:
            self.ksize = tuple(ksize)
        self.sigmas = sigmas
        self.n_kernels = len(self.sigmas)

    def __call__(self, shape, dtype=None):  #, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        
        if dtype is None:
            dtype = 'float32'  # floatx()
        # kernel_size = self.params['ksize']
        if shape is None:
            kernel_size = self.ksize
        else:
            kernel_size = tuple(shape[:2])
            assert self.n_kernels == shape[-1]
#         assert kernel_size == shape[0] == shape[1]
#         sigma = self.params['sigma']

        # Method 1
#         # create nxn zeros
#         inp = np.zeros((kernel_size, kernel_size))
#         # set element at the middle to one, a dirac delta
#         inp[kernel_size//2, kernel_size//2] = 1
#         # gaussian-smooth the dirac, resulting in a gaussian filter mask
#         return low_pass_filter(inp, std)[:,:,0]
        
        # Method 2
        kernels = []
        for sigma in self.sigmas:  # self.params['sigmas']:
            kern_x = cv2.getGaussianKernel(kernel_size[0], sigma)
            kern_y = cv2.getGaussianKernel(kernel_size[1], sigma)
            kern = np.outer(kern_x, kern_y)
            kern /= np.sum(kern)
            kern = kern.astype('float32')  # HACK
            # kern = kern.astype(dtypes.as_dtype(dtype))
            # kern = kern.astype(dtype)
            kern = K.expand_dims(kern, -1)
            kernels.append(kern)
        return K.stack(kernels, axis=-1)
#         return kern

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        # return self.params
        return {'ksize': self.ksize,
                'sigmas': self.sigmas}


class KernelInitializer(Initializer):
    """Kernel initializer class for Conv2D layers."""

    def __init__(self, params, kernel_function, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)
        self.params = params
        self.function = kernel_function
        # self.n_kernels = self.calc_n_kernels(params)
    
    def calc_n_kernels(params):
        return np.prod([value for param, value in params.items() if param != 'ksize'], dtype=int)
#         return len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
#                                  * len(params['gammas']) * len(params['psis'])

    def __call__(self, shape, dtype=None, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        return self.function(**self.params)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        return self.params
        