import os
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


def substitute_layer(model, params, filter_type='gabor', replace_layer=1, 
                     input_shape=None, colour_input='rgb', use_initializer=False, verbose=0):

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
            x = inp
            print(f"{config['shape']}")
        elif ind == replace_layer and params is not None:  # Replace convolutional layer
            print(f"Replacing layer {ind}: '{layer.name}' --> '{filter_type}_conv'...")
            if use_initializer:
                n_kernels = len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
                                              * len(params['gammas']) * len(params['psis'])
                
                # When using this layer as the first layer in a model, provide the keyword argument 
                # input_shape (tuple of integers, does not include the batch axis), 
                # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

                # Input shape: (batch, rows, cols, channels)
                # Output shape: (batch, new_rows, new_cols, filters)
                x = Conv2D(n_kernels, params['ksize'], padding='same', 
                           activation='relu', use_bias=True,
                        #    activation=None, use_bias=False,
                           name=f"{filter_type}_conv",
                           kernel_initializer=GaborInitializer(params))(x)
                # x = Conv2D(n_kernels, params['ksize'], padding='same', activation='relu', use_bias=True)(x)

            else:
                assert isinstance(layer, tf.keras.layers.Conv2D)
                # Generate Gabor filters
                # tensor = get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)
                tensor = get_gabor_tensor(**params)
                x = Lambda(convolve_tensor, arguments={'kernel_tensor': tensor},
                           name=f"{filter_type}_conv")(x)
        elif ind == replace_layer + 1 and params is not None:  # Replace next layer
            # Check input_shape matches output_shape?
            # x = Conv2D(**layers[layer].get_config())(x)
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)
        else:
            # x = layer(x)
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)
        # print(x.shape)

    if use_initializer:
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
