#!/usr/bin/env python

# dtype = 'float16'  # Not supported on Titan Xp
# tf.keras.backend.set_floatx(dtype)
# tf.keras.backend.set_epsilon(1e-4)  # Default 1e-7

import os
import sys
import argparse
import pprint
import warnings
import functools
import csv
import json
from datetime import datetime, timedelta
import time

import numpy as np
import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend
from tensorflow.keras.metrics import categorical_crossentropy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #, load_img, img_to_array
# from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.applications import vgg16, vgg19, resnet_v2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# sys.path.append('/work/generalisation-humans-DNNs/code')
# sys.path.append('/work/generalisation-humans-DNNs/code/accuracy_evaluation/')
# sys.path.append('/work/code/keras_lr_finder/')
# from mappings import HumanCategories
from GaborNet import utils, plots
from GaborNet.preparation import (#as_perturbation_fn, as_greyscale_perturbation_fn, 
                                  get_perturbations,
                                  cifar_wrapper, get_noise_preprocessor, 
                                  sanity_check,
                                  uniform_noise, salt_and_pepper_noise, 
                                  high_pass_filter, low_pass_filter,
                                  adjust_contrast, scramble_phases,
                                  rotate_image, adjust_brightness, 
                                  invert_luminance)

# pprint.pprint(sys.path)
print('+' * 80)  # Simulation metadata
print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Starting simulation...')
print("\nTensorFlow:", tf.__version__)
print(f"Channel ordering: {tf.keras.backend.image_data_format()}")  # TensorFlow: Channels last order.
gpus = tf.config.experimental.list_physical_devices('GPU')
pprint.pprint(gpus)

# warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "tensorflow:Model failed to serialize as JSON.", Warning)


# Instantiate the parser
parser = argparse.ArgumentParser()

# TODO: Check if abbreviations only work for flags
# parser.add_argument('--model', type=str, default='GaborNet',
#                     help='Name of model to use')
parser.add_argument('--convolution', type=str, default='Original',
                    help='Name of convolutional filter to use')
parser.add_argument('--base', type=str, default='VGG16',
                    help='Name of model to use')
parser.add_argument('--pretrain', action='store_true', # type=bool, default=False,
                    help='Flag to use ImageNet weights the model')
parser.add_argument('--architecture', type=str, default='model.json',
                    help='Parameter file (JSON) to load')
parser.add_argument('--upscale', action='store_true', #default=False, required=False,
                    help='Flag to upscale the CIFAR10 images')
parser.add_argument('--interpolate', action='store_true', default=False, required=False,
                    help='Flag to interpolate the images when upscaling')
parser.add_argument('--optimizer', type=str, default='RMSprop',
                    choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                    help='Name of optimizer to use: https://keras.io/optimizers/')
parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, required=False,  # '-r',  
                    help='Learning rate for training')
parser.add_argument('--decay', type=float, default=1e-6, required=False,
                    help='Optimizer decay for training')
parser.add_argument('--use_initializer', action='store_true', default=False, required=False,
                    help='Flag to use the weight initializer (then freeze weights) for the Gabor filters')
parser.add_argument('--trial', type=int, default=1,  # default to 0 when unspecified?
                    help='Trial number for labeling different runs of the same model')
parser.add_argument('--label', type=str, default='',
                    help='For labeling different runs of the same model')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed to use')
parser.add_argument('-t', '--train', action='store_true', # type=bool, default=False,
                    help='Flag to train the model')
# This accompanying unfinished code was deleted in a single commit 25/6/20
# parser.add_argument('--train_with_noise', action='store_true',
#                     help='Flag to train the model with noise-like masks')
parser.add_argument('--recalculate_statistics', action='store_true',
                    help='Flag to recalculate normalisation statistics over the training set')
parser.add_argument('--epochs', type=int, default=20, required=False,
                    help='Number of epochs to train model')
parser.add_argument("--batch", type=int, default=64,
	                help="Size of mini-batches passed to the network")
parser.add_argument('--image_path', type=str, default='',
                    help='Path to image files to load')
parser.add_argument('--train_image_path', type=str, default='',
                    help='Path to training image files to load')
# parser.add_argument('--test_image_path', type=str, default='',
#                     help='Path to testing image files to load')
parser.add_argument('--test_generalisation', action='store_true',
                    help='Flag to test the model on sets of untrained images')
parser.add_argument('--invert_test_images', type=bool, default=True, #action='store_true',
                    help='Flag to invert the luminance of the test images')
parser.add_argument('--test_perturbations', action='store_true',
                    help='Flag to test the model on perturbed images')
parser.add_argument('--data_augmentation', action='store_true', # type=bool, default=False,
                    help='Flag to train the model with data augmentation')
parser.add_argument('-c', '--clean', action='store_true', default=False, required=False,
                    help='Flag to retrain model')
parser.add_argument('--skip_test', action='store_true',
                    help='Flag to skip testing the model')
parser.add_argument('-l', '--log', action='store_true', default=False, required=False,  # type=bool, 
                    help='Flag to log training data')
parser.add_argument('--save_images', action='store_true', default=False, required=False,
                    help='Flag to save preprocessed (perturbed) test images')
parser.add_argument('-p', '--save_predictions', action='store_true', default=False, required=False,  # type=bool, 
                    help='Flag to save category predictions')
parser.add_argument('--gpu', type=int, default=0, required=False,
                    help='GPU ID to run on')

args = vars(parser.parse_args())  # vars() returns a dict

gpus = tf.config.experimental.list_physical_devices('GPU')
assert 0 <= args["gpu"] < len(gpus)
tf.config.experimental.set_visible_devices(gpus[args["gpu"]], 'GPU')

convolution = args['convolution']
base = args['base']
upscale = args['upscale']
interpolate = args['interpolate']
train = args['train']
clean = args['clean']
epochs = args['epochs']
batch = args['batch']  # 64  # 32
image_path = args['image_path']  # Deprecate?
train_image_path = args['train_image_path']
# test_image_path = args['test_image_path']
test_generalisation = args['test_generalisation']
invert_test_images = args['invert_test_images']
test_perturbations = args['test_perturbations']
data_augmentation = args['data_augmentation']
recalculate_statistics = args['recalculate_statistics']
optimizer = args['optimizer']  # 'RMSprop'
lr = args['lr']  # 0.0001  # 0.0005  # 0.0004  # 0.001  # 0.025
decay = args['decay']  # 1e-6  #
use_initializer = args['use_initializer']
skip_test = args['skip_test']
save_images = args['save_images']
save_predictions = args['save_predictions']
seed = args['seed']  # 420420420
# mod = args["model"]
base = args['base']
trial = args['trial']
label = args['label']
assert 0 < trial

# Stimuli metadata
luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma transform)
data_set = 'CIFAR10'
n_classes = 10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
# CIFAR10 image statistics calculated across the training set (after converting to greyscale)
mean = 122.61930353949222
std = 60.99213660091195
colour = 'grayscale'  # 'rgb'
contrast_level = 1  # Proportion of original contrast level for uniform and salt and pepper noise

weights = None

if convolution.capitalize() == 'Gabor':
    # Gabor parameters
    params = {# 'ksize': (127, 127), 
              'ksize': (63, 63),
              'gammas': [0.5], 
    #           'bs': np.linspace(0.4, 2.6, num=3),  # 0.4, 1, 1.8, 2.6
    #           'bs': np.linspace(0.4, 2.6, num=5),
              'bs': np.linspace(1, 2.6, num=3).tolist(),
    #           'bs': np.linspace(1, 2.6, num=5),
    #           'sigmas': [4, 8, 16],  # , 32 
              'sigmas': [8],
              'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
              'psis': [np.pi/2, 3*np.pi/2]}
    mod = f'Gabor_{base}'
elif convolution.capitalize() == 'Low-pass':
    params = {'ksize': (63, 63),
#               'sigmas': [8]
              'sigmas': [1, 2, 4, 8]}
    mod = f'Low-pass_{base}'
elif convolution.capitalize() == 'Original':
    params = None
    mod = base
    if args['pretrain']:
        weights = 'imagenet'
        mod = f'{base}_ImageNet'
else:
    warnings.warn(f'Unknown convolution type: {convolution}!')
    sys.exit()

filter_params = params


max_queue_size = 10
workers = 12  # 4
use_multiprocessing = False
verbose = False
report = 'batch'  # 'epoch'
# use_initializer = False
extension = 'h5'  # For saving model/weights

data_dir = '/work/data'
# Output paths
models_dir = '/work/models'
logs_dir = '/work/logs'
results_dir = '/work/results'
os.makedirs(models_dir, exist_ok=True)
# label is ignored if empty
save_to_dir = os.path.join('/work/results/', label)
os.makedirs(save_to_dir, exist_ok=True)

if save_predictions:
    os.makedirs(os.path.join(save_to_dir, 'predictions'), exist_ok=True)

if save_images:
    image_out_dir = os.path.join(save_to_dir, 'img')
    os.makedirs(image_out_dir, exist_ok=True)
else:
    image_out_dir = None
    image_prefix = ''

print('=' * 80)

# Hardcode noise levels
n_levels = 11
noise_types = get_perturbations(n_levels=n_levels)
# noise_types = [("Uniform", uniform_noise, np.linspace(0, 1, n_levels)),
#                ("Salt and Pepper", salt_and_pepper_noise, np.linspace(0, 1, n_levels)),
#             #    ("High Pass", high_pass_filter, np.logspace(np.log10(5), np.log10(0.3), n_levels)),
#                ("High Pass", high_pass_filter, np.logspace(2, 0, n_levels)),
#             #    ("Low Pass", low_pass_filter, np.logspace(0, np.log10(40), n_levels)),
#                ("Low Pass", low_pass_filter, np.logspace(0, 2, n_levels)),
#                ("Contrast", adjust_contrast, np.logspace(0, -2, n_levels)),
#                ("Phase Scrambling", scramble_phases, np.linspace(0, 180, n_levels)),
#                ("Darken", adjust_brightness, np.linspace(0, -1, n_levels)),
#                ("Brighten", adjust_brightness, np.linspace(0, 1, n_levels)),
#                ("Rotation", rotate_image, np.array([0, 90, 180, 270], dtype=int)),
#                ('Invert', invert_luminance, np.array([0, 1], dtype=int))]

# Process stimuli
if upscale:
    image_size = (224, 224)
    image_shape = image_size + (1,)
    # image_shape = (224, 224, 1)
else:
    image_size = (32, 32)
    image_shape = image_size + (1,)
    # image_shape = (32, 32, 1)

interpolation = cv2.INTER_LANCZOS4  # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC

if image_path and os.path.isdir(image_path):
    load_images_from_disk = True
    # Assumes there are the directories "train" and "test" in image_path

    train_images, x_train, y_train = utils.load_images(os.path.join(image_path, 'train'))
    assert n_classes == len(train_images)

    test_images, x_test, y_test = utils.load_images(os.path.join(image_path, 'test'))  # test_path
    assert n_classes == len(test_images)

# if test_image_path and os.path.isdir(test_image_path):

    
# Set up stimuli
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # RGB format
x_train = np.expand_dims(np.dot(x_train, luminance_weights), axis=-1)
x_test = np.expand_dims(np.dot(x_test, luminance_weights), axis=-1)
y_train = to_categorical(y_train, num_classes=n_classes, dtype='uint8')
y_test = to_categorical(y_test, num_classes=n_classes, dtype='uint8')

# print('-' * 80)
if upscale:
    if interpolate:
        print(f'Interpolating upscaled images with "{interpolation}"...')
        x_train_interp = np.zeros(shape=(x_train.shape[0], *image_shape), dtype=np.float16)
        for i, image in enumerate(x_train):
            x_train_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
        del x_train
        x_train = x_train_interp
        x_train[x_train < 0] = 0
        x_train[x_train > 255] = 255

        x_test_interp = np.zeros(shape=(x_test.shape[0], *image_shape), dtype=np.float16)
        for i, image in enumerate(x_test):
            x_test_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
        del x_test
        x_test = x_test_interp        
        x_test[x_test < 0] = 0
        x_test[x_test > 255] = 255
    else:
        # Equivalent to cv2.INTER_NEAREST (or PIL.Image.NEAREST)
        x_train = x_train.repeat(7, axis=1).repeat(7, axis=2)
        x_test = x_test.repeat(7, axis=1).repeat(7, axis=2)

# NOTE: This is later overridden by the ImageDataGenerator which has 'float32' as the default
x_train = x_train.astype(np.float16)
x_test = x_test.astype(np.float16)

# Summarise stimuli
print(f'x_train.shape: {x_train.shape}')
print(f'Training: {x_train.shape[0]} in {y_train.shape[1]} categories')
print(f'Testing: {x_test.shape[0]} in {y_test.shape[1]} categories')

if data_set == 'CIFAR10' and colour == 'grayscale':
    if (interpolation == cv2.INTER_NEAREST) or not interpolate:
        mean = 122.61930353949222
        std = 60.99213660091195
    elif interpolation == cv2.INTER_LANCZOS4:
        # Without clipping
        # mean = 122.6172103881836
        # std = 60.89457321166992
        # After clipping
        mean = 122.61385345458984
        std = 60.87860107421875
    else:
        print(f'Uncached interpolation method: {interpolation}')
        recalculate_statistics = True
else:
    recalculate_statistics = True

if recalculate_statistics:  # or interpolate:
    print('Recalculating training image statistics...')
    data_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    data_gen.fit(x_train)
    mean = np.squeeze(data_gen.mean).tolist()
    std = np.squeeze(data_gen.std).tolist()
print(f'Training statistics: mean={mean}; std={std}')

# Save metadata
sim = {
    'data_set': data_set,
    'n_classes': n_classes,
    'train': train,
    'epochs': epochs,
    'optimizer': optimizer,
    'lr': lr,
    'decay': decay,
    'batch': batch,
    'data_augmentation': data_augmentation,
    'seed': seed,
    'trial': trial,
    'model': mod,
    'convolution': convolution,
    'base': base,
    'weights': weights,
    'label': label,
    'noise': {noise: levels.tolist() for noise, _, levels in noise_types},
    'image_mean': mean,
    'image_std': std,
    'image_shape': image_shape,
    'upscale': upscale,
    'interpolate': interpolate,
    'interpolation': interpolation,
    'recalculate_statistics': recalculate_statistics,
    'colour': colour,
    'luminance_rgb_weights': luminance_weights.tolist(),
    'contrast_level': contrast_level,
    'save_predictions': save_predictions,
    'image_out_dir': image_out_dir,
    'models_dir': models_dir,
    'results_dir': results_dir,
    'use_initializer': use_initializer,
    'filter_params': params,
    }

# TODO: Replace with f'{conv}_{base}_{trial}'
model_name = f'{mod}_{trial}'
# # sim_set = f"test_{datetime.now().strftime('%Y%m%d')}"
# if label:  # len(label) > 0:
#     sim_set = f"{mod}_{label}_t{trial}_e{epochs}_s{seed}"
# else:
#     sim_set = f"{mod}_t{trial}_e{epochs}_s{seed}"
sim_set = f"{model_name}_s{seed}"
sim_file = f"{sim_set}.json"
with open(os.path.join(save_to_dir, sim_file), "w") as sf:
    json.dump(sim, sf, indent=4)

if save_images:
    stimuli = {
        'noise': {noise: levels.tolist() for noise, _, levels in noise_types},
        'image_mean': mean,
        'image_std': std,
        'image_shape': image_shape,
        'colour': colour,
        'luminance_rgb_weights': luminance_weights.tolist(),
        'contrast_level': contrast_level,
        }
    with open(os.path.join(image_out_dir, 'stimuli.json'), "w") as sf:
        json.dump(stimuli, sf, indent=4)


# for trial in range(start_trial, n_seeds+1):
# seed = start_seed * trial
# for m, mod in enumerate(models):

print('=' * 80)  # Build/load model
print(f"Creating {model_name}...", flush=True)
# Create the model

from all_cnn.networks import allcnn

# get_all_cnn = functools.partial(allcnn, image_shape=image_shape, n_classes=n_classes)

# @functools.wraps(allcnn)
def get_all_cnn(include_top=True, weights=None, input_shape=image_shape, classes=n_classes):
    # model = functools.partial(allcnn, image_shape=image_shape, n_classes=n_classes)
    return allcnn(image_shape=input_shape, n_classes=n_classes)

model_base = {'vgg16': tf.keras.applications.vgg16.VGG16, 
              'vgg19': tf.keras.applications.vgg19.VGG19,
              'resnet': tf.keras.applications.resnet_v2.ResNet50V2,
              'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2, # MobileNetV2
              'inception': tf.keras.applications.inception_v3.InceptionV3,
              'allcnn': get_all_cnn}
# ResNet50, Inception V3, and Xception

# input_tensor = Input(shape=image_shape, name='input_1', dtype='float16')
if weights is None:
    output_classes = n_classes
else:
    output_classes = 1000  # Default
model = model_base[base.lower().replace('-', '')](include_top=True, 
                                                  weights=weights, 
                                                #   input_tensor=input_tensor,
                                                  input_shape=image_shape,
                                                  classes=output_classes)

model = utils.substitute_layer(model, filter_params, 
                               filter_type=convolution,
                               replace_layer=None,
                               input_shape=image_size, 
                               colour_input=colour, 
                               use_initializer=use_initializer)
if n_classes != output_classes:  # 1000:
    model = utils.substitute_output(model, n_classes=n_classes)

opt_args = {'lr': lr, 'decay': decay}
# if optimizer in []:

opt = tf.keras.optimizers.get({'class_name': optimizer, 'config': opt_args})
# opt = tf.keras.optimizers.RMSprop(lr=lr, decay=1e-6)  # CIFAR10

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
if verbose:
    model.summary()

# TODO: Move to new SavedModel format
model_output_dir = os.path.join(models_dir, label, model_name)
os.makedirs(model_output_dir, exist_ok=True)
full_path_to_model = os.path.join(model_output_dir, f"{epochs:03d}_epochs")

print(f"Trial: {trial}; seed={seed}", flush=True)

# if use_initializer or "gabor" not in mod.lower():
# if use_initializer and not ("gabor" in mod.lower() or "low-pass" in mod.lower()):
#     model_data_file = f"{full_path_to_model}_weights.{extension}"
# else:  # Save whole model since metadata can not be saved as JSON
#     model_data_file = f"{full_path_to_model}.{extension}"

model_data_file = f"{full_path_to_model}_weights.{extension}"

if not train:
    print(f"Loading {model_name}...", flush=True)
    model.load_weights(model_data_file)
    print(f"{model_name} loaded!", flush=True)
else:
    # Create Image Data Generators
    if data_augmentation:
        print('Using data augmentation.')
        data_gen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)
    else:
        data_gen = ImageDataGenerator(#preprocessing_function=prep_image,
                                        featurewise_center=True, 
                                        featurewise_std_normalization=True)
    # data_gen.fit(x_train)
    data_gen.mean = mean
    data_gen.std = std
    gen_train = data_gen.flow(x_train, y=y_train, batch_size=batch, 
                                shuffle=True, seed=seed, save_to_dir=None)
    gen_valid = data_gen.flow(x_test, y=y_test, batch_size=batch, 
                                shuffle=True, seed=seed, save_to_dir=None)

    print(f'Checking for {model_data_file}...', flush=True)
    if os.path.exists(model_data_file) and not clean:
        print(f"Found {mod} - skipping training...", flush=True)
        model.load_weights(model_data_file)  # TODO: Check load_weights works when the whole model is saved
        print(f"{model_name} loaded!", flush=True)
    else:
        print(f"Training {mod} for {epochs} epochs...", flush=True)
        t0 = time.time()

        callbacks = []
        if args['log']:
            # Create a tensorboard callback
            # logdir = '/work/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(logs_dir, 'scalars', f'{model_name}-{datetime.now():%Y%m%d-%H%M%S}')
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=5, update_freq='epoch')  # 2048)
            callbacks.append(tensorboard_cb)

        resume_training = False
        csv_logger_cb = tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, f'{model_name}.csv'), 
                                                        append=resume_training, separator=',')
        callbacks.append(csv_logger_cb)

        # Create a callback that saves the model's weights
        # checkpoint_path = os.path.join(models_dir, f"{model_name}.ckpt")
        checkpoint_path = os.path.join(models_dir, "model.ckpt")
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=0)
        callbacks.append(checkpoint_cb)

        save_freq = None  # 10
        if save_freq:
            # checkpoint_path = os.path.join(os.path.join(models_dir, model_name, "{epoch:03d}_epochs.h5"))
            weights_path = os.path.join(os.path.join(model_output_dir, "{epoch:03d}_epochs.h5"))
            # os.makedirs(os.path.join(models_dir, model_name), exist_ok=True)
            weights_cb = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                                            save_weights_only=True,
                                                            verbose=1, period=save_freq)
            callbacks.append(weights_cb)

        reduce_lr_on_plateau = True
        if reduce_lr_on_plateau:
            reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                patience=5, min_lr=1e-8, verbose=1)
            callbacks.append(reduce_lr_cb)

        # Alternative from Geirhos et al. 
        # Set training schedule
        # assert len(boundaries) <= 4
        # boundaries = [-1] * (4 - len(boundaries)) + boundaries
        # print('epoch boundaries for finetuning: {}'.format(boundaries))
        # boundaries = [pretrained_epochs_new + x for x in boundaries]
        # decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

        # def scheduler(epoch):
        #     if epoch < 10:
        #         return 0.001
        #     else:
        #         return 0.001 * tf.math.exp(0.1 * (10 - epoch))
        #         # return lr * 0.5 ** (epoch // 10)
        # def lr_scheduler(epoch):
        #     return learning_rate * (0.5 ** (epoch // lr_drop))

        # scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # history = model.fit_generator(gen_train,
        #                             #   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch))),
        #                               steps_per_epoch=gen_train.n//batch,
        #                               epochs=epochs,
        #                               validation_data=gen_valid,
        #                               validation_steps=gen_valid.n//batch,
        #                               shuffle=True,
        #                               callbacks=callbacks,
        #                               max_queue_size=max_queue_size,
        #                               use_multiprocessing=use_multiprocessing,
        #                               workers=workers)
        
        # if resume_training:
        #     initial_epoch = ...
        # else:
        #     initial_epoch = 0
        history = model.fit(gen_train,
                            epochs=epochs,
                            # steps_per_epoch and steps_per_epoch are required due to a regression in TF 2.2
                            # https://github.com/tensorflow/tensorflow/issues/37968
                            # steps_per_epoch=gen_train.n//batch,
                            steps_per_epoch=len(gen_train),
                            callbacks=callbacks,
                            validation_data=gen_valid,
                            # validation_steps=gen_valid.n//batch,
                            validation_steps=len(gen_valid),
                            shuffle=True,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing)

        if use_initializer:
            model.save_weights(f"{full_path_to_model}_weights.{extension}")  # weights only
            # Does not work with lambda layer
            with open(f"{full_path_to_model}.json", "w") as sf:
                sf.write(model.to_json())  # architecture only
        else:
            model.save(f"{full_path_to_model}.{extension}")  # Full model
        with open(os.path.join(model_output_dir, "simulation.json"), "w") as sf:
            json.dump(sim, sf, indent=4)

        learning_curves = os.path.join(logs_dir, f'{model_name}.png')  # f'{mod}_train_CIFAR10_{trial}.png')
        plots.plot_history(history, chance=1/n_classes, filename=learning_curves)
        # Alternative
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,12))
        # plots.plot_loss(history, ax=ax[0])
        # plots.plot_accuracy(history, chance=1/n_classes, ax=ax[1])
        # fig.savefig(f'{mod}_train_CIFAR10_{trial}_old.png')
        # Old method
        # plots.plot_accuracy(history, chance=1/n_classes, 
        #                     filename=f'{mod}_train_CIFAR10_{trial}.png')
        # plots.plot_loss(history, filename=f'{mod}_train_CIFAR10_loss_{trial}.png')
        t_elapsed = time.time() - t0
        print(f"{model_name} training finished [{str(timedelta(seconds=t_elapsed))}]!", flush=True)
print("=" * 80)

if skip_test:
    print("Skipping testing.", flush=True)
    # Clear GPU memory
    tf.keras.backend.clear_session()
    print("=" * 80)
    sys.exit()


all_test_sets = ['line_drawings', 'silhouettes', 'contours', 'scharr']

if isinstance(test_generalisation, str):
    if test_generalisation.lower() == 'all':
        test_sets = all_test_sets
    elif test_generalisation.lower() in all_test_sets:
        test_sets = [test_generalisation.lower()]
    else:
        warnings.warn(f'Unknown generalisation test set: {test_generalisation}!')
        test_sets = []
elif isinstance(test_generalisation, bool):
    if test_generalisation:
        test_sets = all_test_sets
    else:
        test_sets = []
else:
    warnings.warn(f'Unknown generalisation test set type: {test_generalisation} ({type(test_generalisation)})!')
    test_sets = []

if test_generalisation:
    if invert_test_images:
        # test_sets.extend([f'{test_set}_inverted' for test_set in test_sets])
        inversions = [False, True]
    else:
        inversions = [False]

    fieldnames = ['Trial', 'Model', 'Convolution', 'Base', 'Weights',
                  'Set', 'Inverted', 'Loss', 'Accuracy']
    results_file = os.path.join(save_to_dir, f"generalise_{sim_set}.csv")
    with open(results_file, 'w') as results:
        writer = csv.DictWriter(results, fieldnames=fieldnames)
        writer.writeheader()

# if test_image_path and os.path.isdir(test_image_path):
for test_set in test_sets:
    test_image_path = os.path.join(data_dir, test_set)
    assert os.path.isdir(test_image_path)

    for invert in inversions:
        print(f"Testing {model_name} with images from {test_image_path}{' (inverted)' if invert else ''}...", flush=True)
        t0 = time.time()
        rng = np.random.RandomState(seed=seed)

        # NOTE: Generalisation test images are already in [0, 1] so do not rescale before preprocessing
#         if test_set in ['scharr']:
#             rescale = 1/255
#         else:
#             rescale = 1/255  # 1
        rescale = 1/255

        # Old method: create inverted images on the fly
#         if invert:
#             # prep_image = cifar_wrapper(functools.partial(invert_luminance, level=1),
#             #                            rescale=rescale)
#             prep_image = get_noise_preprocessor("Invert", invert_luminance, level=1, rescale=rescale)
#         else:
#             # prep_image = cifar_wrapper(sanity_check, rescale=rescale)
#             prep_image = get_noise_preprocessor("None", rescale=rescale)

        # New method: use the same preprocessor and load pre-inverted images
        prep_image = get_noise_preprocessor("None", rescale=rescale)
        if invert:
            test_image_path = f"{test_image_path}_inverted"
            assert os.path.isdir(test_image_path)

        data_gen = ImageDataGenerator(# rescale=255,
                                      preprocessing_function=prep_image,
                                      featurewise_center=True, 
                                      featurewise_std_normalization=True)

        # data_gen.fit(x_train)  # Set mean and std
#         if invert:
#             data_gen.mean = 255 - mean
#             data_gen.std = std
#         else:
#             data_gen.mean = mean
#             data_gen.std = std
        data_gen.mean = mean
        data_gen.std = std
        
        if save_images:
            # image_prefix = f"{noise.replace(' ', '_').lower()}"
            generalisation_dir = os.path.join(image_out_dir, f"{test_set}{'_inverted' if invert else ''}")
            os.makedirs(generalisation_dir, exist_ok=True)
            # generalisation_prefix = f"L{l_ind+1:02d}"
            generalisation_prefix = ''
        else:
            generalisation_dir = None
            generalisation_prefix = ''

        gen_test = data_gen.flow_from_directory(test_image_path,
                                                target_size=image_size,
                                                color_mode=colour,
                                                batch_size=batch,
                                                shuffle=False, seed=seed,
                                                interpolation=interpolation,
                                                save_to_dir=generalisation_dir, 
                                                save_prefix=generalisation_prefix)

        metrics = model.evaluate(gen_test, 
                                 # steps=gen_test.n//batch,
                                 steps=len(gen_test),
                                 verbose=1,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)

        row = {'Trial': trial, 'Model': mod, 'Convolution': convolution, 'Base': base, 'Weights': weights,
               'Set': test_set, 'Inverted': invert,  # os.path.basename(test_image_path),
               'Loss': metrics[0], 'Accuracy': metrics[1]}
        with open(results_file, 'a') as results:
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writerow(row)

        if train:
            metrics_dict = {metric: score for metric, score in zip(model.metrics_names, metrics)}
            print(f"Evaluation results: {metrics_dict}")
        else:
            print(f"Evaluation results: {metrics}")

        if save_predictions:
            predictions = model.predict(gen_test, 
                                        verbose=1,
                                        # steps=gen_test.n//batch,  # BAD: This skips the remainder of images
                                        steps=len(gen_test),
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing)
            # print(predictions.shape)  # (n_images, n_classes)
            file_name = f"{model_name}_{test_set}{'_inverted' if invert else ''}.csv"
            predictions_file = os.path.join(save_to_dir, 'predictions', file_name)
            # header = [f'p(class={c})' for c in classes]  # range(n_classes)]
            np.savetxt(predictions_file, predictions, delimiter=',', 
                       header=','.join([f'p(class={c})' for c in classes]))
            print(f'Predictions written to: {predictions_file}')

        t_elapsed = time.time() - t0
        print(f"Testing {test_set}{' (inverted)' if invert else ''} images finished! [{t_elapsed:.3f}s]", flush=True)
        print("-" * 80)

    print('Generalisation testing finished!')
if not len(test_sets):
    print('Generalisation testing skipped!')
print("=" * 80)

# Clear GPU memory
# tf.keras.backend.clear_session()
# sys.exit()


if not test_perturbations:
    # Clear GPU memory
    tf.keras.backend.clear_session()
    sys.exit()

# Create testing results files

# test_metrics = {mod: [] for mod in models}
# if save_predictions:
#     test_predictions = []
#     # test_predictions = {mod: [] for mod in models}
# rows = []

fieldnames = ['Trial', 'Model', 'Convolution', 'Base', 'Weights',
              'Noise', 'Level', 'Loss', 'Accuracy']
results_file = os.path.join(save_to_dir, f"perturb_{sim_set}.csv")
with open(results_file, 'w') as results:
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

# TODO: Optionally test (and generate through the ImageDataGenerator) unperturbed images (L0)
for noise, noise_function, levels in noise_types:
    print(f"[{model_name}] Perturbing test images with {noise} noise...")
    print("-" * 80)
    for l_ind, level in enumerate(levels):
        print(f"[{l_ind+1:02d}/{len(levels):02d}] level={float(level):6.2f}: ", end='', flush=True)

        t0 = time.time()
        # t0 = datetime.now()
        rng = np.random.RandomState(seed=seed+l_ind)

        # if noise in ["Uniform", "Salt and Pepper"]:  # Stochastic perturbations
        #     perturbation_fn = functools.partial(noise_function, level, 
        #                                         contrast_level=1, rng=rng)
        # else:  # Deterministic perturbation
        #     perturbation_fn = functools.partial(noise_function, level)

        prep_image = get_noise_preprocessor(noise, noise_function, level, 
                                            contrast_level=contrast_level, 
                                            bg_grey=mean/255, rng=rng)
#         if noise == "Uniform":
#             perturbation_fn = functools.partial(noise_function, width=level, 
#                                                 contrast_level=contrast_level, rng=rng)
#         elif noise == "Salt and Pepper":
#             perturbation_fn = functools.partial(noise_function, p=level, 
#                                                 contrast_level=contrast_level, rng=rng)
#         elif noise == "High Pass" or noise == "Low Pass":
#             perturbation_fn = functools.partial(noise_function, std=level, bg_grey=mean/255)
#         elif noise == "Contrast":
#             perturbation_fn = functools.partial(noise_function, contrast_level=level)    
#         elif noise == "Phase Scrambling":
#             perturbation_fn = functools.partial(noise_function, width=level)
#         elif noise == "Rotation":
#             perturbation_fn = functools.partial(noise_function, degrees=level)
#         elif noise in ["Darken", "Brighten", "Invert"]:
#             perturbation_fn = functools.partial(noise_function, level=level)
#         else:
#             print(f"Unknown noise type: {noise}!")

#         prep_image = cifar_wrapper(perturbation_fn)
        
        
        # TODO: Check this is still deterministic when parallelised
        data_gen = ImageDataGenerator(preprocessing_function=prep_image,
                                        featurewise_center=True, 
                                        featurewise_std_normalization=True,
                                        dtype='float16')
        # data_gen.fit(x_train)  # Set mean and std
        data_gen.mean = mean
        data_gen.std = std

        if save_images:
            # image_prefix = f"{noise.replace(' ', '_').lower()}"
            test_image_dir = os.path.join(image_out_dir, noise.replace(' ', '_').lower())
            os.makedirs(test_image_dir, exist_ok=True)
            image_prefix = f"L{l_ind+1:02d}"
        else:
            test_image_dir = None

        gen_test = data_gen.flow(x_test, y=y_test, batch_size=batch,
                                 shuffle=False, seed=seed,  # True
                                 save_to_dir=test_image_dir, save_prefix=image_prefix)

#         metrics = model.evaluate_generator(gen_test, #steps=gen_test.n//batch,
#                                            max_queue_size=max_queue_size,
#                                            use_multiprocessing=use_multiprocessing,
#                                            workers=workers)
        # This new method has a memory leak
        metrics = model.evaluate(gen_test, 
                                 # steps=gen_test.n//batch,
                                 steps=len(gen_test),
                                 verbose=0,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)
        # print(model.metrics_names)
        # print(f"{mod} metrics: {metrics}")
        t_elapsed = time.time() - t0
        # t_elapsed = datetime.now() - t0
        if train:
            metrics_dict = {metric: score for metric, score in zip(model.metrics_names, metrics)}
            print(f"{metrics_dict} [{t_elapsed:.3f}s]")
        else:
            print(f"{metrics} [{t_elapsed:.3f}s]")

        # test_metrics[mod].append(metrics)

        if save_predictions:
            # predictions = model.predict_generator(gen_test, steps=gen_test.n//batch)
            predictions = model.predict(gen_test, 
                                        verbose=0,
                                        # steps=gen_test.n//batch,
                                        steps=len(gen_test),
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing)
            # test_predictions.append(predictions)
            # print(predictions.shape)
            # test_predictions[mod].append(predictions)
            # TODO
            # with open(predictions_file, 'a') as pred_file:
            #     pred_writer = csv.
            predictions_file = os.path.join(save_to_dir, 'predictions', 
                                            f'{model_name}_{noise.replace(" ", "_").lower()}_L{l_ind+1:02d}.csv')
            # header = [f'p(class={c})' for c in classes] # range(n_classes)]
            np.savetxt(predictions_file, predictions, delimiter=',', 
                       header=','.join([f'p(class={c})' for c in classes]))

        del gen_test
        row = {'Trial': trial, 'Model': mod, 
               'Convolution': convolution, 'Base': base, 'Weights': weights,
               'Noise': noise, 'Level': level,
               'Loss': metrics[0], 'Accuracy': metrics[1]}
        with open(results_file, 'a') as results:
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writerow(row)
        # rows.append(row)
    print("-" * 80)
print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Simulation finished!')
print("=" * 80)
# for m, metric in enumerate(model.metric_names):
#     test_metrics[mod][metric]

# results = pd.DataFrame(rows)

# Clear GPU memory
tf.keras.backend.clear_session()

# plt.figure()
# sns.relplot(x='Level', y='Accuracy', row='Model', col='Noise', data=results)
# plt.figure()
# sns.lineplot(x='Level', y='Accuracy', kind='line', data=results)

# return model, results