"""
This module contains wrappers and perturbation functions for use with Keras ImageDataGenerator objects.
"""

import functools

import numpy as np
from skimage.color import rgb2grey, grey2rgb
from scipy.ndimage.filters import gaussian_filter
from scipy import fftpack as fp
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


_CHANNEL_MEANS = [103.939, 116.779, 123.68]  # BGR

# L = R * 299/1000 + G * 587/1000 + B * 114/1000  # Used by Pillow.Image.convert('L')
_LUMINANCE_MEAN = 123.68 * 0.299 + 116.779 * 0.587 + 103.939 * 0.114  # 117.378639

stochastic_perturbations = ("Uniform", "Salt and Pepper", "Phase Scrambling")
perturbations = ["Uniform", "Salt and Pepper", "High Pass", "Low Pass", 
                 "Contrast", "Phase Scrambling", "Darken", "Brighten", 
                 "Rotation", "Invert"]


def get_directory_generator(image_directory, #preprocessing_function=None, 
                            rescale=1/255,  # invert=False, 
                            mean=None, std=None, 
                            image_size=(224, 224), colour='grayscale', 
                            batch_size=64, shuffle=False, seed=None,
                            interpolation='lanczos',
                            save_to_dir=None, save_prefix=''):

    # CIFAR10 preprocessing function
    # if invert:
    #     prep_func = get_noise_preprocessor("Invert", invert_luminance, level=1, rescale=rescale)
    # else:
    prep_func = get_noise_preprocessor("None", rescale=rescale)
    data_gen = ImageDataGenerator(# rescale=255,
                                preprocessing_function=prep_func,
                                featurewise_center=True, 
                                featurewise_std_normalization=True)

    if interpolation.lower() == 'nearest':  # cv2.INTER_NEAREST):
        mean = 122.61930353949222
        std = 60.99213660091195
    elif interpolation.lower() == 'lanczos':  # cv2.INTER_LANCZOS4:
        mean = 122.61385345458984
        std = 60.87860107421875
    else:
        print(f'Uncached interpolation method: {interpolation}')
        recalculate_statistics = True
    data_gen.mean = mean
    data_gen.std = std
    
    gen_test = data_gen.flow_from_directory(image_directory,
                                            target_size=image_size,
                                            color_mode=colour,
                                            batch_size=batch_size,
                                            shuffle=False, seed=seed,
                                            interpolation=interpolation,
                                            save_to_dir=save_to_dir, 
                                            save_prefix=save_prefix)
    return gen_test


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


def get_perturbations(n_levels=11):

    noise_types = [
        ("Uniform", uniform_noise, np.linspace(0, 1, n_levels)),
        ("Salt and Pepper", salt_and_pepper_noise, np.linspace(0, 1, n_levels)),
        # ("High Pass", high_pass_filter, np.logspace(np.log10(5), np.log10(0.3), n_levels)),
        ("High Pass", high_pass_filter, np.logspace(2, 0, n_levels)),
        # ("Low Pass", low_pass_filter, np.logspace(0, np.log10(40), n_levels)),
        ("Low Pass", low_pass_filter, np.logspace(0, 2, n_levels)),
        ("Contrast", adjust_contrast, np.logspace(0, -2, n_levels)),
        ("Phase Scrambling", scramble_phases, np.linspace(0, 180, n_levels)),
        ("Darken", adjust_brightness, np.linspace(0, -1, n_levels)),
        ("Brighten", adjust_brightness, np.linspace(0, 1, n_levels)),
        ("Rotation", rotate_image, np.array([0, 90, 180, 270], dtype=int)),
        ('Invert', invert_luminance, np.array([0, 1], dtype=int))
    ]
    return noise_types


def get_noise_preprocessor(name, function=None, level=None, contrast_level=1,
                           bg_grey=None, rng=None, rescale=1/255):

    if name == "Uniform":
        perturbation_fn = functools.partial(function, width=level, 
                                            contrast_level=contrast_level, rng=rng)
    elif name == "Salt and Pepper":
        perturbation_fn = functools.partial(function, p=level, 
                                            contrast_level=contrast_level, rng=rng)
    elif name in ["High Pass", "Low Pass"]:
        perturbation_fn = functools.partial(function, std=level, bg_grey=bg_grey)
    elif name == "Contrast":
        perturbation_fn = functools.partial(function, contrast_level=level)    
    elif name == "Phase Scrambling":
        perturbation_fn = functools.partial(function, width=level, rng=rng)
    elif name == "Rotation":
        perturbation_fn = functools.partial(function, degrees=level)
    elif name in ["Darken", "Brighten", "Invert"]:
        perturbation_fn = functools.partial(function, level=level)
    elif name == "None":  # or function is None:
        perturbation_fn = sanity_check
    else:
        print(f"Unknown noise type: {name}!")

    return cifar_wrapper(perturbation_fn, rescale=rescale)


# Build a perturbed stimulus generator
def cifar_wrapper(f, rescale=1/255):
    """
    Wrapper for perturbation functions when used with upscaled CIFAR10 images.
    
    First rescale the images to the range [0, 1], apply the perturbation, 
    ensure the result is rank 3 (even with a singleton channels dimension),
    set to float32 then rescale to teh range [0, 255].
    
    Args:
        f (function): The perturbation function manipulating images in the range [0, 1].
    
    Returns:
        function: The wrapped perturbation function.
    
    Examples:
        >>> cifar_wrapper(uniform_noise)
        <function bionet.preparation.uniform_noise(image, width, contrast_level, rng)>
    
        Note: To set noise levels, it a partial function must first be created to pass to this wrapper:
        >>> import functools
        >>> functools.partial(uniform_noise, width=3, 
        >>>                   contrast_level=0.8, rng=42)
        functools.partial(<function uniform_noise at 0x7f3876e5a200>, width=3, contrast_level=0.8, rng=42)
    
        >>> import functools
        >>> f = functools.partial(uniform_noise, width=3, 
        >>>                       contrast_level=0.8, rng=42)
        >>> cifar_wrapper(f)
        <function bionet.preparation.cifar_wrapper.<locals>.wrapper(image, *, width=3, contrast_level=0.8, rng=42)>
    """
    @functools.wraps(f)
    def wrapper(image):
        # TODO: Centre and crop image by monkey patching
        # keras_preprocessing.image.utils.loag_img
        # https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905

        # Assume the image has already been converted to greyscale by the ImageDatagenerator: .flow(colour_mode='grayscale')
        assert image.shape == (224, 224, 1), f"Given: {image.shape}"
        # Commented out for speed
        # assert 0 <= np.amin(image)
        # assert np.amax(image) <= 255
        # assert image.dtype == np.float32  # Default for img_to_array keras_preprocessing/image/utils.py#L78
        assert image.dtype in [np.float16, np.float32, np.float64]
        # Standard ImageNet preprocessing but set mode='torch' for [0, 1] scaling
        # allowing the perturbation functions from image_manipulation.py to be used

        # Convert to greyscale - handled by perturbations in image_manipulation.py
        # image = rgb2grey(image)

        # Scale to [0, 1]
        # image /= 255
        # print(np.amin(image), np.amax(image), flush=True)
        image *= rescale
        # pad = 0.02  # Allow some tolerance in unprocessed images
        # assert 0 - pad <= np.amin(image), f"Min = {np.amin(image)}"
        # assert np.amax(image) <= 1 + pad, f"Max = {np.amax(image)}"
        # assert np.amax(image) > 0.5
        assert 0 <= np.amin(image), f"Min = {np.amin(image)}"
        assert np.amax(image) <= 1, f"Max = {np.amax(image)}"
        # image[image < 0] = 0
        # image[image > 1] = 1

        # Assume this expects 2D arrays (224, 224) of RGB values in the range [0, 1]
        perturbed = f(np.squeeze(image))

        assert 0 <= np.amin(perturbed), f"Min = {np.amin(perturbed)}"
        assert np.amax(perturbed) <= 1, f"Max = {np.amax(perturbed)}"

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
        # perturbed /= rescale
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
    image = (1-contrast_level)/2.0 + image.dot(contrast_level)
    image[image < 0] = 0
    image[image > 1] = 1
    return image


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
    # np.clip(image, 0, 1, out=image)
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

    # Handle type conversion since correlate1d uses np.float64:
    # https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py
    
    # apply the gaussian filter and subtract from the original image
    # gauss_filter = gaussian_filter(image[:, :, 0], std, mode='constant', cval=bg_grey)
    gauss_filter = gaussian_filter(image.astype(np.float64), std, mode='constant', cval=bg_grey)
    new_image = image - gauss_filter.astype(image.dtype)#[..., np.newaxis]

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

    # Handle type conversion since correlate1d uses np.float64:
    # https://github.com/scipy/scipy/blob/master/scipy/ndimage/filters.py
    
    # covert image to greyscale and define variable prepare new image
    # image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # apply Gaussian low-pass filter
    # new_image = gaussian_filter(image[:, :, 0], std, mode='constant', cval=bg_grey)
    new_image = gaussian_filter(image.astype(np.float64), std, mode='constant', cval=bg_grey)
    new_image = new_image.astype(image.dtype)
#     new_image = gaussian_filter(image.astype(np.float32), std, output=new_image, mode='constant', cval=bg_grey)
    # new_image = new_image[..., np.newaxis]

    # gaussian_filter(image, std, output=new_image, mode='constant', cval=bg_grey)

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image,new_image,new_image))


def phase_scrambling(image, width, rng=None):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""

    return scramble_phases(image, width, rng)


def power_equalisation(image, avg_power_spectrum):
    """Equalise images' power spectrum by setting an image's amplitudes 
    in the Fourier domain to the amplitude average over all used images.
    
    parameter:
    - image: a numpy.ndarray"""

    return equalise_power_spectrum(image, avg_power_spectrum)


def scramble_phases(image, width, rng=None):
    """Apply random shifts to an images' frequencies' phases in the Fourier domain.
    
    parameter:
    - image: an numpy.ndaray
    - width: maximal width of the random phase shifts"""

    # create array with random phase shifts from the interval [-width,width]
    length = (image.shape[0]-1)*(image.shape[1]-1)
    if rng is None:
        phase_shifts = np.random.random(length//2) - 0.5
    else:
        phase_shifts = rng.random(length//2) - 0.5
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
