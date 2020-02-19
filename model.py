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

sys.path.append('/work/generalisation-humans-DNNs/code')
sys.path.append('/work/generalisation-humans-DNNs/code/accuracy_evaluation/')
sys.path.append('/work/code/keras_lr_finder/')
from mappings import HumanCategories
from GaborNet import utils, plots
from GaborNet.preparation import (as_perturbation_fn, as_greyscale_perturbation_fn, 
                                  cifar_wrapper, sanity_check,
                                  uniform_noise, salt_and_pepper_noise, 
                                  high_pass_filter, low_pass_filter,
                                  adjust_contrast, scramble_phases,
                                  rotate_image)

# pprint.pprint(sys.path)
print("\nTensorFlow:", tf.__version__)
print(f"Channel ordering: {tf.keras.backend.image_data_format()}")  # TensorFlow: Channels last order.
# print("RGB/BGR")
gpus = tf.config.experimental.list_physical_devices('GPU')
pprint.pprint(gpus)

# warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "tensorflow:Model failed to serialize as JSON.", Warning)


# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, default='GaborNet',
                    help='Name of model to use')
parser.add_argument('-a', '--architecture', type=str, default='model.json',
                    help='Parameter file (JSON) to load')
parser.add_argument('--upscale', action='store_true', #default=False, required=False,
                    help='Flag to upscale the CIFAR10 images')
parser.add_argument('--interpolate', action='store_true', default=False, required=False,
                    help='Flag to interpolate the images when upscaling')
parser.add_argument('-o', '--optimizer', type=str, default='RMSprop',
                    choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                    help='Name of optimizer to use: https://keras.io/optimizers/')
parser.add_argument('-r', '--lr', '--learning_rate', type=float, default=1e-4, required=False,
                    help='Learning rate for training')
parser.add_argument('-d', '--decay', type=float, default=1e-6, required=False,
                    help='Optimizer decay for training')
parser.add_argument('-i', '--use_initializer', action='store_true', default=False, required=False,
                    help='Flag to use the weight initializer (then freeze weights) for the Gabor filters')
parser.add_argument('--trial', type=int, default=1,  # default to 0 when unspecified?
                    help='Trial number For labeling different runs of the same model')
parser.add_argument('--label', type=str, default='',
                    help='For labeling different runs of the same model')
parser.add_argument('-s', '--seed', type=int, default=42,
                    help='Random seed to use')
parser.add_argument('-t', '--train', action='store_true', # type=bool, default=False,
                    help='Flag to train the model')
parser.add_argument('--train_with_noise', action='store_true',
                    help='Flag to train the model with noise-like masks')
parser.add_argument('--recalculate_statistics', action='store_true',
                    help='Flag to recalculate normalisation statistics over the training set')
parser.add_argument('--epochs', type=int, default=20, required=False,
                    help='Number of epochs to train model')
parser.add_argument("-b", "--batch", type=int, default=64,
	                help="Size of mini-batches passed to the network")
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

upscale = args['upscale']
interpolate = args['interpolate']
train = args['train']
clean = args['clean']
epochs = args['epochs']
batch = args['batch']  # 64  # 32
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
mod = args["model"]
trial = args['trial']
label = args['label']
assert 0 < trial

# Stimuli metadata
luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma transform)
data_set = 'CIFAR10'
n_classes = 10
# CIFAR10 image statistics calculated across the training set (after converting to greyscale)
mean = 122.61930353949222
std = 60.99213660091195
colour = 'grayscale'  # 'rgb'
contrast_level = 1  # Proportion of original contrast level for uniform and salt and pepper noise

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

# params = None

max_queue_size = 10
workers = 12  # 4
use_multiprocessing = False
verbose = False
report = 'batch'  # 'epoch'
# use_initializer = False
extension = 'h5'  # For saving model/weights

# Output paths
models_dir = '/work/models'
logs_dir = '/work/logs'
results_dir = '/work/results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

if save_images:
    save_to_dir = '/work/results/img/'
    if label:  # len(label) > 0:
        save_to_dir = os.path.join(save_to_dir, label)
    os.makedirs(save_to_dir, exist_ok=True)
else:
    save_to_dir = None
    save_prefix = ''

# Hardcode noise levels
n_levels = 11
noise_types = [("Uniform", uniform_noise, np.linspace(0, 0.5, n_levels)),
               ("Salt and Pepper", salt_and_pepper_noise, np.linspace(0, 0.5, n_levels)),
            #    ("High Pass", high_pass_filter, np.logspace(np.log10(5), np.log10(0.3), n_levels)),
            #    ("High Pass", high_pass_filter, np.logspace(0, -1, n_levels)),
               ("High Pass", high_pass_filter, np.logspace(2, 0, n_levels)),
            #    ("Low Pass", low_pass_filter, np.logspace(0, np.log10(40), n_levels)),
            #    ("Low Pass", low_pass_filter, np.logspace(-1, 1, n_levels)),
               ("Low Pass", low_pass_filter, np.logspace(0, 2, n_levels)),
               ("Contrast", adjust_contrast, np.logspace(0, -2, n_levels)),
               ("Phase Scrambling", scramble_phases, np.linspace(0, 180, n_levels))]#,
               #("Rotation", rotate_image, np.array([0, 90, 180, 270], dtype=int))]

if upscale:
    image_size = (224, 224)
    image_shape = image_size + (1,)
    # image_shape = (224, 224, 1)
else:
    image_size = (32, 32)
    image_shape = image_size + (1,)
    # image_shape = (32, 32, 1)

interpolation = cv2.INTER_LANCZOS4  # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC

# TODO: Finish or remove
if args['train_with_noise']:
    # Train the model with the noise-like masks
    data_augmentation = False
    data_root = '/work/data/pixel/large'
    stimulus_set = 'static'
    noise_types = ['Single-pixel']  # ['Original', 'Salt-and-pepper', 'Additive', 'Single-pixel']
    test_conditions = ['Same', 'Diff', 'Orig']


    data_path = os.path.join(data_root, stimulus_set, noise_type.replace('-', '_').lower())  # 'single_pixel')
    train_path = os.path.join(data_path, 'train')

    if os.path.isfile(os.path.join(train_path, 'x_train.npy')) and \
        os.path.isfile(os.path.join(train_path, 'y_train.npy')) and not clean:
        print(f'Loading {data_set}:{stimulus_set} data arrays.')
        x_train = np.load(os.path.join(train_path, 'x_train.npy'))
        y_train = np.load(os.path.join(train_path, 'y_train.npy'))
        cat_dirs = [os.path.join(train_path, o) for o in os.listdir(train_path)
                    if os.path.isdir(os.path.join(train_path, o))]
        assert n_classes == len(cat_dirs)
    else:
        print(f'Loading {data_set}:{stimulus_set} image files.')
        train_images, x_train, y_train = utils.load_images(train_path)
        print(train_images.keys())
        assert n_classes == len(train_images)
        np.save(os.path.join(train_path, 'x_train.npy'), x_train)
        np.save(os.path.join(train_path, 'y_train.npy'), y_train)

    test_sets = []
    for test_cond in test_conditions:
        test_path = os.path.join(data_path, f"test_{test_cond.lower()}")
        if os.path.isfile(os.path.join(test_path, 'x_test.npy')) and \
            os.path.isfile(os.path.join(test_path, 'y_test.npy')) and not clean:
            x_test = np.load(os.path.join(test_path, 'x_test.npy'))
            y_test = np.load(os.path.join(test_path, 'y_test.npy'))
        else:
            test_images, x_test, y_test = load_images(test_path)
            print(test_images.keys())
            assert n_classes == len(test_images)
            np.save(os.path.join(test_path, 'x_test.npy'), x_test)
            np.save(os.path.join(test_path, 'y_test.npy'), y_test)
        test_sets.append((x_test, y_test))
    test_cond = "Orig"  # Use this for examining learning curves
    x_test, y_test = test_sets[test_conditions.index("Orig")]  # Unpack default test set
    # else:
    #     sys.exit(f"Unknown data set requested: {data_set}")
else:
    # Set up stimuli
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # RGB format
    x_train = np.expand_dims(np.dot(x_train, luminance_weights), axis=-1)
    x_test = np.expand_dims(np.dot(x_test, luminance_weights), axis=-1)
    y_train = to_categorical(y_train, num_classes=n_classes, dtype='uint8')
    y_test = to_categorical(y_test, num_classes=n_classes, dtype='uint8')

if upscale:
    if interpolate:
        print(f'Interpolating upscaled images with "{interpolation}"...')
        x_train_interp = np.zeros(shape=(x_train.shape[0], *image_shape), dtype=np.float16)
        for i, image in enumerate(x_train):
            x_train_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
        del x_train
        x_train = x_train_interp

        x_test_interp = np.zeros(shape=(x_test.shape[0], *image_shape), dtype=np.float16)
        for i, image in enumerate(x_test):
            x_test_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
        del x_test
        x_test = x_test_interp
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

if recalculate_statistics or interpolate:
    data_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    data_gen.fit(x_train)
    mean = data_gen.mean
    std = data_gen.std
print(f'Training statistics: mean={mean}; std={std}')

# Save metadata
sim = {'classes': n_classes,
       'train': train,
       'epochs': epochs,
       'optimizer': optimizer,
       'lr': lr,
       'decay': decay,
       'batch': batch,
       'seed': seed,
       'trial': trial,
       'model': mod,
       'label': label,
       'noise': {noise: levels.tolist() for noise, _, levels in noise_types},
       'image_mean': mean,
       'image_std': std,
       'image_shape': image_shape,
       'colour': colour,
       'luminance_rgb_weights': luminance_weights.tolist(),
       'contrast_level': contrast_level,
       'image_out_dir': save_to_dir,
       'models_dir': models_dir,
       'results_dir': results_dir,
       'use_initializer': use_initializer,
       'filter_params': params,
      }

# sim_set = f"test_{datetime.now().strftime('%Y%m%d')}"
if label:  # len(label) > 0:
    sim_set = f"{mod}_{label}_t{trial}_e{epochs}_s{seed}"
else:
    sim_set = f"{mod}_t{trial}_e{epochs}_s{seed}"
sim_file = f"{sim_set}.json"
with open(os.path.join(results_dir, sim_file), "w") as sf:
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
    with open(os.path.join(save_to_dir, 'stimuli.json'), "w") as sf:
        json.dump(stimuli, sf, indent=4)


# for trial in range(start_trial, n_seeds+1):
# seed = start_seed * trial
# for m, mod in enumerate(models):

model_name = f'{mod}_{trial}'
print(f"Creating {model_name}...")
# Create the model
if mod == "GaborNet":  #  and params is not None:
    weights = None
    filter_params = params
elif mod.endswith("ImageNet"):
    weights = 'imagenet'
    filter_params = None
else:
    weights = None
    filter_params = None
model = tf.keras.applications.vgg16.VGG16(weights=weights, 
                                            include_top=True, 
                                            classes=1000)
model = utils.substitute_layer(model, filter_params, 
                                input_shape=image_size, 
                                colour_input=colour, 
                                use_initializer=use_initializer)
if n_classes != 1000:
    model = utils.substitute_output(model, n_classes=n_classes)

opt_args = {'lr': lr, 'decay': decay}
# if optimizer in []:

opt = tf.keras.optimizers.get({'class_name': optimizer, 'config': opt_args})
# opt = tf.keras.optimizers.RMSprop(lr=lr, decay=1e-6)  # CIFAR10

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
if verbose:
    model.summary()

# model_output_dir = os.path.join(models_dir, f"{epochs:03d}_epochs")
model_output_dir = os.path.join(models_dir, label, model_name)
os.makedirs(model_output_dir, exist_ok=True)
# full_path_to_model = os.path.join(model_output_dir, model_name)
full_path_to_model = os.path.join(model_output_dir, f"{epochs:03d}_epochs")

print(f"Trial: {trial}; seed={seed}")

if not train:
    print(f"Loading {model_name}...")
    model.load_weights(full_path_to_model)  # .h5
    print(f"{model_name} loaded!")
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

    # if "gabor" in mod.lower() and not use_initializer:
    if use_initializer or "gabor" not in mod.lower():
        model_data_file = f"{full_path_to_model}.{extension}"
    else:
        model_data_file = f"{full_path_to_model}_weights.{extension}"

    # if os.path.exists(f"{full_path_to_model}.index") and not clean:
    if os.path.exists(f"{full_path_to_model}.{extension}") and not clean:  # .index
        # if os.path.exists(model_data_file) and not clean
        print(f"Found {mod} - skipping training...")
        model.load_weights(f'{full_path_to_model}.{extension}')
        print(f"{model_name} loaded!")
    else:
        print(f"Training {mod} for {epochs} epochs...")
        t0 = time.time()

        callbacks = []
        if args['log']:
            # Create a tensorboard callback
            # logdir = '/work/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(logs_dir, 'scalars', f'{datetime.now():%Y%m%d-%H%M%S}')
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq=2048)
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

        history = model.fit_generator(gen_train,
                                    #   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch))),
                                      steps_per_epoch=gen_train.n//batch,
                                      epochs=epochs,
                                      validation_data=gen_valid,
                                      validation_steps=gen_valid.n//batch,
                                      shuffle=True,
                                      callbacks=callbacks,
                                      max_queue_size=max_queue_size,
                                      use_multiprocessing=use_multiprocessing,
                                      workers=workers)

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
        print(f"{model_name} training finished [{str(timedelta(seconds=t_elapsed))}]!")
print("=" * 80)

if skip_test:
    print("Skipping testing.")
    # Clear GPU memory
    tf.keras.backend.clear_session()
    print("=" * 80)
    sys.exit()


# Create testing results files

# test_metrics = {mod: [] for mod in models}
if save_predictions:
    test_predictions = []
    # test_predictions = {mod: [] for mod in models}
rows = []

fieldnames = ['Trial', 'Model', 'Noise', 'Level', 'Loss', 'Accuracy']
results_file = os.path.join(results_dir, f"{sim_set}.csv")
with open(results_file, 'w') as results:
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()


# TODO: Finish or remove
if args['train_with_noise']:

    if label:  # len(label) > 0:
        sim_set = f"{mod}_{label}_t{trial}_e{epochs}_s{seed}"
    else:
        sim_set = f"{mod}_t{trial}_e{epochs}_s{seed}"
    sim_file = f"{sim_set}.json"
    with open(os.path.join(results_dir, sim_file), "w") as sf:
        json.dump(sim, sf, indent=4)

    rows = []

    fieldnames = ['Trial', 'Model', 'Noise', 'Level', 'Loss', 'Accuracy']
    results_file = os.path.join(results_dir, f"{sim_set}.csv")
    with open(results_file, 'w') as results:
        writer = csv.DictWriter(results, fieldnames=fieldnames)
        writer.writeheader()

    # cond_acc = {}
    # cond_loss = {}
    for test_cond, (x_test, y_test) in zip(test_conditions, test_sets):

        metrics = model.evaluate(x=x_test, y=y_test, batch_size=batch)

        row = {'Trial': trial, 'Model': mod, 'Noise': noise, 'Level': level,
                'Loss': metrics[0], 'Accuracy': metrics[1]}
        with open(results_file, 'a') as results:
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writerow(row)
        rows.append(row)

    # print("Saving metrics: ", model.metrics_names)
    # with open(os.path.join(save_dir, f'{model_name}_CONDVALACC.json'), "w") as jf:
    #     json.dump(cond_acc, jf)
    # if save_loss:
    #     with open(os.path.join(save_dir, f'{model_name}_CONDVALLOSS.json'), "w") as jf:
    #         json.dump(cond_loss, jf)

# TODO: Optionally test (and generate through the ImageDataGenerator) unperturbed images (L0)
for noise, noise_fuction, levels in noise_types:
    print(f"[{model_name}] Perturbing test images with {noise} noise...")
    print("-" * 80)
    for l_ind, level in enumerate(levels):
        print(f"[{l_ind+1:02d}/{len(levels)}] level={float(level):6.2f}: ", end='', flush=True)

        t0 = time.time()
        # t0 = datetime.now()
        rng = np.random.RandomState(seed=seed+l_ind)

        # if noise in ["Uniform", "Salt and Pepper"]:  # Stochastic perturbations
        #     perturbation_fn = functools.partial(noise_fuction, level, 
        #                                         contrast_level=1, rng=rng)
        # else:  # Deterministic perturbation
        #     perturbation_fn = functools.partial(noise_fuction, level)

        if noise == "Uniform":
            perturbation_fn = functools.partial(noise_fuction, width=level, 
                                                contrast_level=contrast_level, rng=rng)
        elif noise == "Salt and Pepper":
            perturbation_fn = functools.partial(noise_fuction, p=level, 
                                                contrast_level=contrast_level, rng=rng)
        elif noise == "High Pass" or noise == "Low Pass":
            perturbation_fn = functools.partial(noise_fuction, std=level)
        elif noise == "Contrast":
            perturbation_fn = functools.partial(noise_fuction, contrast_level=level)    
        elif noise == "Phase Scrambling":
            perturbation_fn = functools.partial(noise_fuction, width=level)
        elif noise == "Rotation":
            perturbation_fn = functools.partial(noise_fuction, degrees=level)
        else:
            print(f"Unknown noise type: {noise}!")

        prep_image = cifar_wrapper(perturbation_fn)
        # TODO: Check this is still deterministic when parallelised
        data_gen = ImageDataGenerator(preprocessing_function=prep_image,
                                        featurewise_center=True, 
                                        featurewise_std_normalization=True)
        # data_gen.fit(x_train)  # Set mean and std
        data_gen.mean = mean
        data_gen.std = std

        if save_images:
            # save_prefix = f"{noise.replace(' ', '_').lower()}"
            test_image_dir = os.path.join(save_to_dir, noise.replace(' ', '_').lower())
            os.makedirs(test_image_dir, exist_ok=True)
            save_prefix = f"L{l_ind+1:02d}"
        else:
            test_image_dir = None

        gen_test = data_gen.flow(x_test, y=y_test, batch_size=batch,
                                    shuffle=False, seed=seed,  # True
                                    save_to_dir=test_image_dir, save_prefix=save_prefix)
                                    # save_to_dir=save_to_dir, save_prefix=save_prefix)

        metrics = model.evaluate_generator(gen_test, steps=gen_test.n//batch,
                                            max_queue_size=max_queue_size,
                                            use_multiprocessing=use_multiprocessing,
                                            workers=workers)
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
            predictions = model.predict_generator(gen_test, steps=gen_test.n//batch)
            test_predictions.append(predictions)
            print(predictions.shape)
            # test_predictions[mod].append(predictions)
            # TODO
            # with open(predictions_file, 'a') as pred_file:
            #     pred_writer = csv.
        row = {'Trial': trial, 'Model': mod, 'Noise': noise, 'Level': level,
                'Loss': metrics[0], 'Accuracy': metrics[1]}
        with open(results_file, 'a') as results:
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writerow(row)
        rows.append(row)
    print("-" * 80)
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