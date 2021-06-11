import os
import glob
import functools
import subprocess
import math
from pprint import pprint

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
# import PIL.Image
import seaborn as sns
import numpy as np
from scipy import signal, fftpack
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import auc
from scipy.integrate import simps

from bionet.utils import find_conv_layer, calc_sigma, calc_lambda  # calc_bandwidth,
from bionet.preparation import (perturbations, cifar_wrapper, sanity_check, 
                                invert_luminance, get_noise_preprocessor)
from bionet.config import (convolutions_order, bases_order, classes, 
                           n_classes, luminance_weights)


# luminance_weights = np.array([0.299, 0.587, 0.114])

def inspect_image(image_path):
    """Plot image with Luminance Histogram, Cumulative Luminance Histogram 
    and log absolute of values of Fourier Spectrum."""
    
    figsize = (8,8)
    
    if isinstance(image_path, str):
        try:
            print(subprocess.check_output(["identify", image_path]))
        except OSError as e:
            if e.errno == errno.ENOENT:
                # handle file not found error.
                print(f"File not found: {image_path}")
                return
            else:
                # Something else went wrong while trying to run command
                print("Program 'identify' not found!")
                raise
        image = plt.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        print("Error: Unknown image type!")

    if image.ndim > 2:
        if image.shape[-1] == 1:
            image = np.squeeze(image)
            n_channels = 1
        else:
            n_channels = image.shape[-1]
    else:
        n_channels = 1
    print(f"Image shape: {image.shape}; channels: {n_channels}; "
          f"scaling: [{np.amin(image)}, {np.amax(image)}]; "
          f"mean={np.mean(image):.3}; std={np.std(image):.3}")
    
    # img = PIL.Image.open(image_path)
    # exif_data = img._getexif()
    # print(exif_data)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    immap = axes[0, 0].imshow(image, cmap="gray")
    plt.colorbar(immap, ax=axes[0, 0])
    axes[0, 0].set_title("Image")
    
    if n_channels == 3:
        image = np.dot(image, luminance_weights)
    fft2 = fftpack.fft2(image)
    fftmap = axes[1, 0].imshow(np.log10(np.abs(fft2)))
    plt.colorbar(fftmap, ax=axes[1, 0])
    axes[1, 0].set_title("log10(abs(FFT2))")

    # TODO: Plot histograms for each channel if there are more than one
    # See: https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935
    bins = 128
    axes[0, 1].hist(image.ravel(), bins=bins)
    axes[0, 1].set_title("Luminance Histogram")
    
    axes[1, 1].hist(image.ravel(), bins=bins, cumulative=True)
    axes[1, 1].set_title("Cumulative Luminance Histogram")


def plot_accuracy(history, chance=None, filename=None, ax=None, figsize=(12, 8)):
    """Plot accuracy against training epochs with optional validation accuracy
    and chance level."""
    # Plot training metrics
        
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = len(history.history['acc'])
    # for metric in history.history:
    #     ax.plot(range(1, epochs+1), history.history[metric], label=metric)
    ax.plot(range(1, epochs+1), history.history['acc'], label="Training")
    if 'val_acc' in history.history:
        ax.plot(range(1, epochs+1), history.history['val_acc'], label="Validation")
    if chance:
        ax.axhline(y=chance, color='grey', linestyle='--', label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, epochs+1))
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0, 1))
    ax.legend()
    if filename:
        fig.savefig(filename)
    return (fig, ax)


def plot_loss(history, filename=None, ax=None, figsize=(12, 8)):
    """Plot loss against training epochs with optional validation loss."""
    # Plot training metrics
        
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = len(history.history['loss'])
    # for metric in history.history:
    #     ax.plot(range(1, epochs+1), history.history[metric], label=metric)
    ax.plot(range(1, epochs+1), history.history['loss'], label="Training")
    if 'val_loss' in history.history:
        ax.plot(range(1, epochs+1), history.history['val_loss'], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, epochs+1))
    ax.set_ylabel("Loss")
    # ax.set_ylim((0, 1))
    ax.legend()
    if filename:
        fig.savefig(filename)
    return (fig, ax)


def plot_metrics(history, epochs, metrics, ax):
    """Plot all recorded metrics."""
    for metric in metrics:
        ax.plot(range(1, epochs+1), history.history[metric], label=metric)
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, epochs+1))
    ax.set_ylabel("Score")
    ax.legend()


def plot_history(history, chance=None, metrics=None, filename=None, figsize=(12, 16)):
    """Plot training history with optional chance accuracy level."""

    if metrics is None:
        metrics = list(history.history.keys())
    
    acc_metrics = []
    loss_metrics = []
    other_metrics = []

    for metric in metrics:
        if 'acc' in metric:
            acc_metrics.append(metric)
        elif 'loss' in metric:
            loss_metrics.append(metric)
        else:
            other_metrics.append(metric)
    
    include_acc = bool(len(acc_metrics) > 0)
    include_loss = bool(len(loss_metrics) > 0)
    include_other = bool(len(other_metrics) > 0)

    epochs = len(history.history['loss'])  # Always included in history?

    nrows = int(include_acc) + int(include_loss) + int(include_other)
    assert nrows > 0
    if figsize is None:
        width, height = 8, 3
        figsize = (width, height*nrows)

    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex=True, squeeze=True, figsize=figsize)

    row = 0
    if include_loss:
        plot_metrics(history, epochs, loss_metrics, ax[row])
        ax[row].set_ylabel("Loss")
        if include_acc or include_other:
            ax[row].set_xlabel('')
        row += 1

    if include_acc:
        plot_metrics(history, epochs, acc_metrics, ax[row])
        if chance:
            ax[row].axhline(y=chance, color='grey', linestyle='--', label="Chance")
            # ax[row].hlines(chance, 1, epochs, color='grey', linestyle='--', label="Chance")
        ax[row].set_ylabel("Accuracy")
        ax[row].set_ylim((0, 1))
        if include_other:
            ax[row].set_xlabel('')
        row += 1
    
    if include_other:
        plot_metrics(history, epochs, other_metrics, ax[row])
        ax[row].set_ylabel("Metric")
        row += 1

    if filename:
        fig.savefig(filename)
    return (fig, ax)


def plot_model_kernels(model, layers=None, fig_sf=1):

    if layers is None:
        layer_name, layer_ind = find_conv_layer(model, first=True)
        layers = [layer_ind]

    results = {}
    for layer_ind in layers:
        layer = model.layers[layer_ind]
        assert isinstance(layer, tf.keras.layers.Conv2D)

        filters, biases = layer.get_weights()
        print(layer_ind, layer.name, filters.shape)

        n_channels_in = filters.shape[-2]
        n_channels_out = filters.shape[-1]
        fig, axes = plt.subplots(nrows=n_channels_in, ncols=n_channels_out, sharex=True, sharey=True, squeeze=False, figsize=(n_channels_out*fig_sf, n_channels_in*fig_sf))
        for c_in in range(n_channels_in):
            for c_out in range(n_channels_out):
                axes[c_in, c_out].imshow(filters[:, :, c_in, c_out])
                axes[c_in, c_out].set_xticks([])
                axes[c_in, c_out].set_yticks([])
                if c_in == n_channels_in - 1:
                    axes[c_in, c_out].set_xlabel(c_out)
                if c_out == 0:
                    axes[c_in, c_out].set_ylabel(c_in)
        results[layer.name] = (filters, biases)
    return results


def plot_gabor_filters(params, images=None, use_gpu=True, fontsize=20, space=0.15, verbose=0):
    """Plot a bank of Gabor filters decribed and labelled by their parameters.
    Optionally pass an image or list of images to convolve with each filter."""
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
        print(f"Wavelength range: [{lambda_min:#.3g}, {lambda_max:#.3g}] pixels")  # Hex?
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


def plot_image_predictions(label, models, top_k=3, params=None, gpu=None):
    # TODO: Remove convolution steps and plot original image instead. 
    # label = 'long'
    # models = [f'Low-pass_VGG19_{n}' for n in range(1, 2)]

    
    # params = {'ksize': (63, 63),
    #           'sigmas': [8]}

    test_sets = ['line_drawings', 'silhouettes', 'contours', 'scharr']
    # luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma
    n_samples = 10
    # n_classes = 10
    rescale = 1/255
    fig_sf = 2


    def get_predictions(file):
        # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        #            'dog', 'frog', 'horse', 'ship', 'truck']
        probabilities = np.loadtxt(file, delimiter=',')
        rankings = np.argsort(probabilities, axis=1)
        predictions = []
        for r_image, p_image in zip(rankings, probabilities):
            # Descending order [::-1]
            predictions.append({classes[c]: p_image[c] for c in reversed(r_image)})
        return predictions

    if params is not None:
        if gpu is not None:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

        kernel_size = params['ksize']
        kernels = []
        for sigma in params['sigmas']:
            kern_x = cv2.getGaussianKernel(kernel_size[0], sigma)
            kern_y = cv2.getGaussianKernel(kernel_size[1], sigma)
            kern = np.outer(kern_x, kern_y)
            kern /= np.sum(kern)
            kern = kern.astype('float32')  # HACK
            if gpu is not None:
                kern = K.expand_dims(kern, -1)
            kernels.append(kern)

    for test_set in test_sets:
        root = f'/work/data/{test_set}/'

        for invert_test_images in [False, True]:
            if invert_test_images:
    #             prep_image = cifar_wrapper(functools.partial(invert_luminance, level=1), rescale=1)
                prep_image = get_noise_preprocessor("Invert", invert_luminance, level=1, rescale=rescale)
                annotated_test_set = f'{test_set}_inverted'
            else:
    #             prep_image = cifar_wrapper(sanity_check, rescale=1)
                prep_image = get_noise_preprocessor("None", rescale=rescale)
                annotated_test_set = f'{test_set}'

            for model in models:
                predictions = get_predictions(f'/work/results/{label}/predictions/{model}_{annotated_test_set}.csv')

                fig, axes = plt.subplots(nrows=n_classes, ncols=n_samples, 
                                         figsize=(fig_sf*n_samples, fig_sf*n_classes))
                i = 0
                for c_ind, category in enumerate(sorted(os.listdir(root))):
                    for s_ind, img in enumerate(sorted(os.listdir(os.path.join(root, category)))):
                        if annotated_test_set.startswith('scharr') and s_ind >= 10:
                            break
                        full_path = os.path.join(root, category, img)
                #         image = np.squeeze(prep_image(plt.imread(full_path)[..., np.newaxis]*255))/255
                        image = plt.imread(full_path) * 255
                        if image.shape == (224, 224):
                            image = image[..., np.newaxis]
                        if image.shape == (224, 224, 3):
                            image = np.dot(image, luminance_weights)[..., np.newaxis]
                        image = np.squeeze(prep_image(image))

                        if params is not None:
                            if gpu is None:
                                fimg = signal.convolve2d(image, kernels[0], mode='same')
                            else:
                                kernels = K.stack(kernels, axis=-1)
                                kernels = K.expand_dims(kernels, -1)
                                kernels = K.expand_dims(kernels, -1)
                                image = K.expand_dims(image, -1)
                                fimg = K.conv2d(image, kernels, padding='same')
                                fimg = fimg.numpy().squeeze()

                            image = fimg

                        axes[c_ind, s_ind].imshow(image, cmap='gray', vmin=0, vmax=255)
                        axes[c_ind, s_ind].set_xticks([])
                        axes[c_ind, s_ind].set_yticks([])
                        axes[c_ind, s_ind].get_xaxis().set_visible(False)
                        axes[c_ind, s_ind].get_yaxis().set_visible(False)
                        axes[c_ind, s_ind].set_axis_off()

                        annotations = []
                        for c_label, prob in list(predictions[i].items())[:top_k]:
                            annotations.append(f'{c_label}: {prob:04.1%}')
                        if annotations[0].startswith(category):
                            colour = 'g'
                        else:
                            colour = 'r'
                        bound = {'fc': colour, 'boxstyle': 'round,pad=.5', 'alpha': 0.5}
                        axes[c_ind, s_ind].text(216, 8, '\n'.join(annotations), 
                                                va='top', ha='right', 
                                                fontsize='xx-small', bbox=bound)

                        if c_ind == 0:
                            axes[s_ind, c_ind].set_ylabel(category.capitalize())
                        i += 1

                # plt.axis('off')
                fig.subplots_adjust(0.02,0.02,0.98,0.98)
                output = f'/work/results/{label}/filtered_{test_set}{"_invert" if invert_test_images else ""}_{model}.png'
                fig.savefig(output)
                f"Created figure: {output}"


def plot_perturbations(df, tag=None, wrap=5, fig_sf=2, verbose=0):  # landscape=True, 
    """Plot a grid of performance versus level curves for a battery of noise perturbations."""

#     noise_types = ["Uniform", "Salt and Pepper", "High Pass", "Low Pass", 
#                    "Contrast", "Phase Scrambling", "Darken", "Brighten", 
#                    "Rotation", "Invert"]
    noise_types = perturbations
    chance_level = 1 / n_classes

    models = df.Model.unique().tolist()
    convolutions = df.Convolution.unique().tolist()
    bases = df.Base.unique().tolist()

    if verbose:
        print(models)
        print(convolutions)
        print(bases)

    # with sns.axes_style("ticks"):
    with sns.axes_style("darkgrid"):

#         if landscape:
#             ncols = wrap
#             nrows = int(math.ceil(len(noise_types)/ncols))
#             # fig, axes = plt.subplots(ncols, int(math.ceil(len(noise_types)/ncols)),
#             #                          sharey=True, squeeze=True, figsize=(24,12))
#         else:  # portrait
#             nrows = wrap
#             ncols = int(math.ceil(len(noise_types)/nrows))

        ncols = wrap
        nrows = int(math.ceil(len(noise_types)/ncols))
        fig, axes = plt.subplots(nrows, ncols, sharey=True, squeeze=True, 
                                 figsize=(ncols * fig_sf, nrows * fig_sf))
        
        for (noise, ax) in zip(noise_types, axes.ravel()):
            ax = sns.lineplot(x='Level', y='Accuracy',
                              style='Base', style_order=bases_order,
                              hue='Convolution', hue_order=convolutions_order,
                              data=df[df.Noise==noise], ax=ax)
            if ax.get_legend():
                ax.get_legend().set_visible(False)
            ax.set_title(noise)
            ax.set_ylim([0, 1])
            ax.set_yticks(np.linspace(0, 1, num=11))
            ax.axhline(y=chance_level, c='grey', ls='--')

            if noise in ["High Pass", "Low Pass", "Contrast"]:
                ax.set_xscale('log')
                # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
                ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, num=9)))
                if noise in ["High Pass", "Contrast", "Darken"]:
                    ax.invert_xaxis()

            if noise in ["Phase Scrambling", "Rotation", "Invert"]:
                x = df.query(f"Noise == '{noise}' and Model == '{models[0]}' and Trial == 1").Level.to_numpy()
                x = sorted(list(set(x)))  # Get unique values
                ax.set_xticks(x)
                # print(x)
        
        if ax.get_legend():
            ax.get_legend().set_visible(True)  # Show in final subfigure
        plt.tight_layout()
        if tag is not None:
            fig.savefig(f'/work/results/{tag}/perturbations.png')

    return fig, axes


def plot_auc_ratios(df, tag=None):

    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(20,6))
        ax = sns.barplot(data=df, x='Noise', y='Ratio', hue='Model')
#         sns.catplot(data=df, x='Noise', y='Ratio', hue='Model', kind='bar', aspect=2)
    #     ax.axhline(y=0, c='k')
        ax.axhline(y=1, c='gray', ls='--')
    #     sns.despine(trim=True, bottom=True)  # offset=10, 
        plt.tight_layout()
        if tag is not None:
            fig.savefig(f'/work/results/auc_{tag}.png')

    return fig, ax


def inspect_results(label, seed="*", reference="VGG16", annotate="Convolution", save=False, 
                    fig_sf=3, results_dir="/work/results", reference_dir="paper", verbose=0):

    # Perturbation results
    columns = ['Trial', 'Model', 'Convolution', 'Base', 'Weights', 
               'Noise', 'Level', 'Loss', 'Accuracy']

    # if seed is None:
    #     file = "perturb_*.csv"
    # else:
    #     file = f"perturb_*s{seed}.csv"
    pattern = os.path.join(results_dir, label, f"perturb_*s{seed}.csv")
    if isinstance(reference, (list, tuple)):
        references = [os.path.join(results_dir, reference_dir, f"perturb_{ref}_Set.csv") for ref in reference]
    elif isinstance(reference, str):
        references = [os.path.join(results_dir, reference_dir, f"perturb_{reference}_Set.csv")]  # Compare to standard VGG-16 results
    else:
        print(f"Unknown reference_set passed: {reference} ({type(reference)})")


    df = pd.concat([pd.read_csv(filename) for filename in glob.glob(pattern)], ignore_index=True)
    # if "imagenet" in filename.lower():
    #     df["Weights"] = "imagenet"
    # else:
    #     df["Weights"] = "None"
    df["Weights"] = df["Weights"].fillna("None")
    df = df[columns]
    df.sort_values(by=['Trial', 'Noise', 'Level'], inplace=True, ignore_index=True)
    if save:
        output = os.path.join(results_dir, label, f"perturb_{label}_Set.csv")
        df.to_csv(output, index=False)  # Save new results
    # df["Model"] = "*" + df["Model"]  # Preprend "*" to differentiate new results
    # df["Base"] = "*" + df["Base"]  # Preprend "*" to differentiate new results
    # df["Convolution"] = "*" + df["Convolution"]  # Preprend "*" to differentiate new results
    if annotate:  # Preprend "*" to differentiate new results for plots
        df[annotate] = "*" + df[annotate]
    perturbations = pd.concat([df, *[pd.read_csv(ref) for ref in references]], ignore_index=True)

    fig, axes = plot_perturbations(perturbations, label, fig_sf=fig_sf)

    if verbose:
        print(perturbations.pivot_table(values='Accuracy', columns='Noise', index=['Model'],
                                        aggfunc=len, fill_value=0, margins=True))


    # Generalisation results
    aspect = 1.25  # width = height * aspect

    pattern = os.path.join(results_dir, label, f"generalise_*_s{seed}.csv")
    # output = os.path.join(results_dir, label, "generalise_Set.csv")
    reference = os.path.join(results_dir, reference_dir, "generalise_Set.csv")  # Compare to standard results

    df = pd.concat([pd.read_csv(filename) for filename in glob.glob(pattern)], ignore_index=True)
    df.sort_values(by=['Trial', 'Convolution', 'Base'], inplace=True, ignore_index=True)  # , 'Set', 'Inverted'
    # df.to_csv(output, index=False)  # Save new results
    # generalisations = pd.concat([pd.read_csv(output), pd.read_csv(reference)], ignore_index=True)
    # df["Base"] = "*" + df["Base"]  # Preprend "*" to differentiate new results
    # df["Convolution"] = "*" + df["Convolution"]  # Preprend "*" to differentiate new results
    if annotate:  # Preprend "*" to differentiate new results for plots
        df[annotate] = "*" + df[annotate]
    generalisations = pd.concat([df, pd.read_csv(reference)], ignore_index=True)

    # hue_order = ['VGG16', 'VGG19', 'ALL-CNN']
    models = generalisations.Base.unique().tolist()
    hue_order = sorted(models)

    # sns.set_context("talk")
    with sns.axes_style("darkgrid"):
        g = sns.catplot(x='Convolution', y='Accuracy', col='Set', row='Inverted', hue='Base', hue_order=hue_order, 
                        data=generalisations, kind='bar', height=fig_sf, aspect=aspect)
        g.set(ylim=(0, 1))
        for ax in g.axes.ravel():
            ax.axhline(y=1/n_classes, c='grey', ls='--')
        g.savefig(os.path.join(results_dir, f"{label}_pred_accuracy.png"))

    if verbose:
        print(generalisations.pivot_table(values='Accuracy', columns=['Set', 'Inverted'], 
                                          index=['Model'], aggfunc=len, fill_value=0, margins=True))

    return perturbations, generalisations
