import functools

import numpy as np
from skimage.color import rgb2grey, grey2rgb
from tensorflow.keras.applications.vgg19 import preprocess_input


_CHANNEL_MEANS = [103.939, 116.779, 123.68]

def as_perturbation_fn(f):
    @functools.wraps(f)
    def wrapper(image):  # , colour_input=True):

        # TODO: Centre and crop image by monkey patching
        # keras_preprocessing.image.utils.loag_img
        # https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905

        assert image.shape == (224, 224, 3)
        assert image.dtype == np.float32
        # Standard ImageNet preprocessing but set mode='torch' for [0, 1] scaling
        # allowing the perturbation functions from image_manipulation.py to be used

        # Convert to greyscale - handled by perturbations in image_manipulation.py
        # image = rgb2grey(image)

        # TODO: Move out?
        image = preprocess_input(image, mode='caffe')


        image += _CHANNEL_MEANS
        image /= 255
        image = image[..., ::-1]  # Assumes channels last
        # Assume this expects RGB values in range [0, 1]
        perturbed = f(image)
        
        # TODO: Try converting back
        # image = grey2rgb(image)

        # perturbed *= 255
        # image = preprocess_input(image, mode='torch')
        # image = preprocess_input(image, mode='caffe')

        assert perturbed.dtype in [np.float32, np.float64]
        # Replicate greyscale values for all three channels
        if perturbed.ndim == 2:
            perturbed = perturbed[..., np.newaxis].repeat(3, axis=2)
        assert image.shape == perturbed.shape
        if perturbed.dtype == np.float64:
            perturbed = perturbed.astype(np.float32)

        # Convert back because VGG19 expects zero-centred BGR images
        perturbed = perturbed[..., ::-1]
        perturbed *= 255
        perturbed -= _CHANNEL_MEANS

        # if not colour_input:
        #     perturbed = rgb2grey(perturbed)
        #     perturbed = perturbed[..., 0]

        return perturbed

    return wrapper


def sanity_check(image):
    return image


def unprocess_input(image):

    image += _CHANNEL_MEANS
    image /= 255
    image = image[..., ::-1]  # Assumes channels last

    return image
