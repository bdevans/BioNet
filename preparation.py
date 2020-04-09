import functools

import numpy as np
from skimage.color import rgb2grey, grey2rgb
from scipy.ndimage.filters import gaussian_filter
from scipy import fftpack as fp
from tensorflow.keras.applications.vgg19 import preprocess_input


_CHANNEL_MEANS = [103.939, 116.779, 123.68]  # BGR

# L = R * 299/1000 + G * 587/1000 + B * 114/1000  # Used by Pillow.Image.convert('L')
_LUMINANCE_MEAN = 123.68 * 0.299 + 116.779 * 0.587 + 103.939 * 0.114  # 117.378639

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


def as_greyscale_perturbation_fn(f):
    @functools.wraps(f)
    def wrapper(image):

        # TODO: Centre and crop image by monkey patching
        # keras_preprocessing.image.utils.loag_img
        # https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905

        # Assume the image has already been converted to greyscale by the ImageDatagenerator
        assert image.shape == (224, 224, 1)
        # Commented out for speed
        # assert 0 <= np.amin(image)
        # assert np.amax(image) <= 255
        assert image.dtype == np.float32  # Default for img_to_array keras_preprocessing/image/utils.py#L78
        # assert image.dtype in [np.float16, np.float32, np.float64]
        # Standard ImageNet preprocessing but set mode='torch' for [0, 1] scaling
        # allowing the perturbation functions from image_manipulation.py to be used

        # Convert to greyscale - handled by perturbations in image_manipulation.py
        # image = rgb2grey(image)

        # TODO: Move out?
        # Scale to [0, 1]
        image /= 255

        # Assume this expects RGB values in range [0, 1]
        perturbed = f(image)

        assert perturbed.dtype in [np.float16, np.float32, np.float64]
        # Replicate greyscale values for all three channels
        if perturbed.ndim == 2:
            perturbed = perturbed[..., np.newaxis]
        assert image.shape == perturbed.shape
        if perturbed.dtype == np.float64:
            perturbed = perturbed.astype(np.float32)

        # Convert back because VGG19 expects zero-centred BGR images
        perturbed *= 255
        # NOTE: This could be passed to ImageDataGenerator with featurewise_center
        # This would allow contrast reduction to be taken into account
        perturbed -= _LUMINANCE_MEAN

        return perturbed

    return wrapper



# Build a perturbed stimulus generator
def cifar_wrapper(f):
    @functools.wraps(f)
    def wrapper(image):
        # TODO: Centre and crop image by monkey patching
        # keras_preprocessing.image.utils.loag_img
        # https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905

        # Assume the image has already been converted to greyscale by the ImageDatagenerator
        assert image.shape == (224, 224, 1)
        # Commented out for speed
        # assert 0 <= np.amin(image)
        # assert np.amax(image) <= 255
        # assert image.dtype == np.float32  # Default for img_to_array keras_preprocessing/image/utils.py#L78
        assert image.dtype in [np.float16, np.float32, np.float64]
        # Standard ImageNet preprocessing but set mode='torch' for [0, 1] scaling
        # allowing the perturbation functions from image_manipulation.py to be used

        # Convert to greyscale - handled by perturbations in image_manipulation.py
        # image = rgb2grey(image)

        # TODO: Move out?
        # Scale to [0, 1]
        image /= 255

        # Assume this expects RGB values in range [0, 1]
        perturbed = f(np.squeeze(image))

        assert perturbed.dtype in [np.float16, np.float32, np.float64]
        if perturbed.ndim == 3 and perturbed.shape[-1] > 1:
            # Should this be np.expand_dims(np.dot(perturbed, luminance_weights), axis=-1)?
            perturbed = perturbed[:, :, 0]
        # Replicate greyscale values for all three channels
        if perturbed.ndim == 2:
            perturbed = perturbed[..., np.newaxis]
        assert image.shape == perturbed.shape, f"Original: {image.shape} --> Perturbed: {perturbed.shape}"
        if perturbed.dtype == np.float64:
            perturbed = perturbed.astype(np.float32)
        # perturbed = perturbed.astype(np.float16)

        # Convert back because VGG19 expects zero-centred BGR images
        perturbed *= 255
        # NOTE: This could be passed to ImageDataGenerator with featurewise_center
        # This would allow contrast reduction to be taken into account
#             perturbed -= _LUMINANCE_MEAN
        return perturbed
    return wrapper


def adjust_contrast(image, contrast_level):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."

    return (1-contrast_level)/2.0 + image.dot(contrast_level)


def salt_and_pepper_noise(image, p, contrast_level, rng, check=False):
    """Convert to grayscale. Adjust contrast. Apply salt and pepper noise.
    parameters:
    - image: a numpy.ndarray
    - p: a scalar indicating probability of white and black pixels, in [0, 1]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    assert 0 <= p <= 1

    # image = grayscale_contrast(image, contrast_level)
    image = adjust_contrast(image, contrast_level)
    # assert image.ndim == 2

    u = rng.uniform(size=image.shape)

    salt = (u >= 1 - p / 2).astype(image.dtype)
    pepper = -(u < p / 2).astype(image.dtype)

    image = image + salt + pepper
    # image = np.clip(image, 0, 1)
    # Faster
    image[image < 0] = 0
    image[image > 1] = 1

    if check:
        assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def uniform_noise(image, width, contrast_level, rng):
    """Convert to grayscale. Adjust contrast. Apply uniform noise.

    parameters:
    - image: a numpy.ndarray 
    - width: a scalar indicating width of additive uniform noise
             -> then noise will be in range [-width, width]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    # image = grayscale_contrast(image, contrast_level)
    image = adjust_contrast(image, contrast_level)

    return apply_uniform_noise(image, -width, width, rng)


def apply_uniform_noise(image, low, high, rng=None, check=False):
    """Apply uniform noise to an image, clip outside values to 0 and 1.

    parameters:
    - image: a numpy.ndarray 
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    nrow = image.shape[0]
    ncol = image.shape[1]

    # Fixed: Using [..., np.newaxis] bevause get_uniform_noise is 2-D but image is 3-D
    image = image + get_uniform_noise(low, high, nrow, ncol, rng)#[..., np.newaxis]

    #clip values
    # image = np.where(image < 0, 0, image)
    # image = np.where(image > 1, 1, image)
    # Faster
    image[image < 0] = 0
    image[image > 1] = 1

    if check:
        assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def get_uniform_noise(low, high, nrow, ncol, rng=None):
    """Return uniform noise within [low, high) of size (nrow, ncol).

    parameters:
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - nrow: number of rows of desired noise
    - ncol: number of columns of desired noise
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    if rng is None:
        return np.random.uniform(low=low, high=high,
                                 size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high,
                           size=(nrow, ncol))


def is_in_bounds(mat, low, high):
    """Return wether all values in 'mat' fall between low and high.

    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all(np.logical_and(mat >= 0, mat <= 1))


def high_pass_filter(image, std, bg_grey=0.4423):
    """Apply a Gaussian high pass filter to a greyscale converted image.
    by calculating Highpass(image) = image - Lowpass(image).
    
    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation"""

    # set this to mean pixel value over all images
#     bg_grey = 0.4423

    # convert image to greyscale and define variable prepare new image
    # image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # apply the gaussian filter and subtract from the original image
    # gauss_filter = gaussian_filter(image[:, :, 0], std, mode='constant', cval=bg_grey)
    gauss_filter = gaussian_filter(image, std, mode='constant', cval=bg_grey)
    new_image = image - gauss_filter#[..., np.newaxis]

    # add mean of old image to retain image statistics
    mean_diff = bg_grey - np.mean(new_image, axis=(0,1))
    new_image = new_image + mean_diff

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image,new_image,new_image))


def low_pass_filter(image, std, bg_grey=0.4423):
    """Aplly a Gaussian low-pass filter to an image.
    
    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    """
    # set this to mean pixel value over all images
#     bg_grey = 0.4423

    # covert image to greyscale and define variable prepare new image
    # image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # apply Gaussian low-pass filter
    # new_image = gaussian_filter(image[:, :, 0], std, mode='constant', cval=bg_grey)
    new_image = gaussian_filter(image, std, mode='constant', cval=bg_grey)
    new_image = new_image#[..., np.newaxis]

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image,new_image,new_image))


def phase_scrambling(image, width):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""

    return scramble_phases(image, width)


def power_equalisation(image, avg_power_spectrum):
    """Equalise images' power spectrum by setting an image's amplitudes 
    in the Fourier domain to the amplitude average over all used images.
    
    parameter:
    - image: a numpy.ndarray"""

    return equalise_power_spectrum(image, avg_power_spectrum)


def scramble_phases(image, width):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""

    # create array with random phase shifts from the interval [-width,width]
    length = (image.shape[0]-1)*(image.shape[1]-1)
    phase_shifts = np.random.random(length//2) - 0.5
    phase_shifts = phase_shifts * 2 * width/180 * np.pi

    # convert to graysclae
    # channel = rgb2grey(image)
    channel = image

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel) #rfft for real values
    f = fp.fftshift(f)

    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # transformations of phases
    # just change the symmetric parts of FFT outcome, which is
    # [1:,1:] for the case of even image sizes
    fnew_phase = f_phase
    fnew_phase[1:, 1:] = shift_phases(f_phase[1:, 1:], phase_shifts)

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = f_amp*np.exp(1j*fnew_phase)

    # reverse shift to centre and perform Fourier Backwards Transformation
    fnew = fp.ifftshift(fnew)
    new_channel = fp.ifft2(fnew)

    # make sure that there are no imaginary parts after transformation
    new_channel = new_channel.real

    # clip too small and too large values
    new_channel[new_channel > 1] = 1
    new_channel[new_channel < 0] = 0

    # return stacked (RGB) grey image
    return np.dstack((new_channel, new_channel, new_channel))


def shift_phases(f_phase, phase_shifts):
    """Applies phase shifts to an array of phases.
    
    parameters: 
    - f_phase: the original images phases (in frequency domain)
    - phase_shifts: an array of phase shifts to apply to the original phases 
    """

    # flatten array for easier transformation
    f_shape = f_phase.shape
    flat_phase = f_phase.flatten()
    length = flat_phase.shape[0]

    # apply phase shifts symmetrically to complex conjugate frequency pairs
    # do not change c-component
    flat_phase[:length//2] += phase_shifts
    flat_phase[length//2+1:] -= phase_shifts

    # reshape into output format
    f_phase = flat_phase.reshape(f_shape)

    return f_phase


def equalise_power_spectrum(image, avg_power_spectrum):
    """Equalise images' power spectrum by setting an image's amplitudes 
    in the Fourier domain to the amplitude average over all used images.
    
    parameter:
    - image: a numpy.ndarray
    - avg_power_spectrum: an array of the same dimension as one of images channels
                          containing the average over all images amplitude spectrum"""

    # check input dimensions
    assert image.shape[:2] == avg_power_spectrum.shape, 'Image shape={} unequal \
            avg_spectrum shape={}'.format(image.shape[:2], avg_power_spectrum.shape)

    # convert image to greysclae
    # channel = rgb2grey(image)
    channel = image

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel)
    f = fp.fftshift(f)

    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # set amplitudes to average power spectrum
    fnew_amp = avg_power_spectrum

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = fnew_amp*np.exp(1j*f_phase)

    # reverse shift to centre and perform Fourier Backwards Transformation
    fnew = fp.ifftshift(fnew)
    new_channel = fp.ifft2(fnew)

    # make sure that there are no imaginary parts after transformation
    new_channel = new_channel.real

    # clip too large and too small values
    new_channel[new_channel > 1] = 1
    new_channel[new_channel < 0] = 0

    # return stacked (RGB) grey image
    return(np.dstack((new_channel, new_channel, new_channel)))


def rotate_image(image, degrees):
    if degrees == 0:
        rotated_image = image
    elif degrees == 90:
        rotated_image = rotate90(image)
    elif degrees == 180:
        rotated_image = rotate180(image)
    elif degrees == 270:
        rotated_image = rotate270(image)
    else:
        print(f"Error: Unsupported rotation: {degrees} degrees!")
        rotated_image = None
    return rotated_image

def rotate90(image):
    """Rotate an image by 90 degrees.

    parameters:
    - image: a numpy.ndarray"""

    # grey_channel = rgb2grey(image)
    grey_channel = image
    new_channel = np.transpose(grey_channel, axes=(1,0))[::-1,:]
    return np.dstack((new_channel,new_channel,new_channel))

def rotate180(image):
    """Rotate an image by 180 degrees.
    
    parameters:
    - image: a numpy.ndarray"""

    # grey_channel = rgb2grey(image)
    grey_channel = image
    new_channel = grey_channel[::-1,::-1]
    return np.dstack((new_channel,new_channel,new_channel))

def rotate270(image):
    """Rotate an image by 270 degrees.
    
    parameters:
    - image: a numpy.ndarray"""

    # grey_channel = rgb2grey(image)
    grey_channel = image
    new_channel = np.transpose(grey_channel[::-1,:], axes=(1,0))
    return np.dstack((new_channel,new_channel,new_channel))


def adjust_brightness(image, level):
    
    new_image = image.copy()
    new_image += level
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1
    return new_image


def invert_luminance(image, level):
    if level < 0.5:
        return image

    new_image = 1 - image
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1
    return new_image
