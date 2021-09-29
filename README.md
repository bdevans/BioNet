# BioNet
Deep Convolutional Neural Networks with bio-inspired filters. 

1. Clone this repository
2. Clone the CIFAR-10G generalisation test set
3. Optionally clone an ALL-CNN implementation
4. Set your `project_dir` in the notebook and pass your `data_dir` (`ln -s /shared/data/ data`)

Expected directory structure
----------------------------

```
.
├── bionet
│   ├── config.py
│   ├── explain.py
│   ├── __init__.py
│   ├── plots.py
│   └── preparation.py
├── data
│   ├── CIFAR-10G
│   ├── ecoset
│   └── ecoset-cifar10
├── logs
├── models
├── notebooks
├── results
├── scripts
├── model.py
└── README.md
```

Training and testing the model
------------------------------

The main script to handle training and testing is `model.py` in the project's root directory. 

### Arguments and usage

```
usage: model.py [-h] [--convolution CONVOLUTION] [--base BASE] [--pretrain]
                [--architecture ARCHITECTURE] [--interpolation INTERPOLATION]
                [--optimizer {SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam}]
                [--lr LR] [--decay DECAY] [--use_initializer]
                [--internal_noise INTERNAL_NOISE] [--trial TRIAL]
                [--label LABEL] [--seed SEED] [-t] [--recalculate_statistics]
                [--epochs EPOCHS] [--batch BATCH] [--image_path IMAGE_PATH]
                [--train_image_path TRAIN_IMAGE_PATH] [--test_generalisation]
                [--invert_test_images INVERT_TEST_IMAGES]
                [--test_perturbations] [--data_augmentation]
                [--extra_augmentation] [-c] [--skip_test] [-l] [--save_images]
                [-p] [--gpu GPU] [--project_dir PROJECT_DIR] [-v VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  --convolution CONVOLUTION
                        Name of convolutional filter to use
  --base BASE           Name of model to use
  --pretrain            Flag to use pretrained ImageNet weights in the model
  --architecture ARCHITECTURE
                        Parameter file (JSON) to load
  --interpolation INTERPOLATION
                        Method to interpolate the images when upscaling.
                        Default: 0 ("nearest" i.e. no interpolation)
  --optimizer {SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam}
                        Name of optimizer to use: https://keras.io/optimizers/
  --lr LR, --learning_rate LR
                        Learning rate for training
  --decay DECAY         Optimizer decay for training
  --use_initializer     Flag to use the weight initializer (then freeze
                        weights) for the Gabor filters
  --internal_noise INTERNAL_NOISE
                        Standard deviation for adding a Gaussian noise layer
                        after the first convolutional layer
  --trial TRIAL         Trial number for labeling different runs of the same
                        model
  --label LABEL         For labeling different runs of the same model
  --seed SEED           Random seed to use
  -t, --train           Flag to train the model
  --recalculate_statistics
                        Flag to recalculate normalisation statistics over the
                        training set
  --epochs EPOCHS       Number of epochs to train model
  --batch BATCH         Size of mini-batches passed to the network
  --image_path IMAGE_PATH
                        Path to image files to load
  --train_image_path TRAIN_IMAGE_PATH
                        Path to training image files to load
  --test_generalisation
                        Flag to test the model on sets of untrained images
  --invert_test_images INVERT_TEST_IMAGES
                        Flag to invert the luminance of the test images
  --test_perturbations  Flag to test the model on perturbed images
  --data_augmentation   Flag to train the model with data augmentation
  --extra_augmentation  Flag to train the model with additional data
                        augmentation
  -c, --clean           Flag to retrain model
  --skip_test           Flag to skip testing the model
  -l, --log             Flag to log training data
  --save_images         Flag to save preprocessed (perturbed) test images
  -p, --save_predictions
                        Flag to save category predictions
  --gpu GPU             GPU ID to run on
  --project_dir PROJECT_DIR
                        Path to the root project directory
  -v VERBOSE, --verbose VERBOSE
                        Verbosity level
```

To train and test the models, the code below may be used and adapted as required.

```python
import os
import sys
import pprint
import subprocess
import random

from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend

project_root_dir = "/home/jovyan/work/BioNet"  # Change as necessary
print(f"Project directory: {project_root_dir}\n")
sys.path.append(project_root_dir)

print("\nTensorFlow:", tf.__version__)
print(f"Channel ordering: {tf.keras.backend.image_data_format()}")  # TensorFlow: Channels last order.
gpus = tf.config.experimental.list_physical_devices('GPU')
# gpus = tf.config.list_physical_devices('GPU')
pprint.pprint(gpus)


label = "paper"
image_path = ''  # Empty string defaults to CIFAR-10
# image_path = '/shared/data/ecoset-cifar10'
convolutions = ['Original', 'Low-pass', 'DoG', 'Gabor', 'Combined-trim']
bases = ['ALL-CNN', 'VGG-16', 'VGG-19', 'ResNet']

seed = 0
start_trial = 1
num_trials = 5
trials = range(start_trial, start_trial+num_trials)
train = True
pretrain = False
clean = False
epochs = 100
optimizer = "RMSprop"
lr = 1e-4
use_initializer = True
data_augmentation = True
extra_augmentation = False
internal_noise = 0
skip_test = False
save_images = False
save_predictions = True
test_generalisation = True
test_perturbations = True
interpolation = 4  # Lanczos
recalculate_statistics = False
verbose = 0
halt_on_error = False
gpu = 1

######################################

script = os.path.join(project_root_dir, "model.py")
flags = ['--log']
if train:
    flags.append('-t')
if clean:
    flags.append('-c')
if use_initializer:
    flags.append('--use_initializer')
if data_augmentation:
    flags.append('--data_augmentation')
if extra_augmentation:
    flags.append('--extra_augmentation')
if skip_test:
    flags.append('--skip_test')
if recalculate_statistics:
    flags.append('--recalculate_statistics')
if save_predictions:
    flags.append('--save_predictions')

optional_args = []
if image_path:
    optional_args.extend(['--image_path', str(image_path)])
if test_perturbations:
    optional_args.append('--test_perturbations')
if test_generalisation:
    optional_args.append('--test_generalisation')
if pretrain:
    optional_args.append('--pretrain')
if internal_noise:
    optional_args.extend(['--internal_noise', str(internal_noise)])
if interpolation:
    optional_args.extend(['--interpolation', str(interpolation)])
if verbose:
    optional_args.extend(['--verbose', str(verbose)])

count = 1
for trial in tqdm(trials, desc='Trial'):
    if seed is None:
        seed = random.randrange(2**32)
    for base in tqdm(bases, desc='Model Base', leave=False):
        for conv in tqdm(convolutions, desc='Convolution', leave=False):
            cmd = [script, *flags]
            if save_images and count == 1:
                cmd.append('--save_images')
            cmd.extend(['--convolution', conv, '--base', base, '--label', label,
                        '--trial', str(trial), '--seed', str(seed),
                        '--optimizer', optimizer, '--lr', str(lr),
                        '--epochs', str(epochs), '--gpu', str(gpu)])
            cmd.extend(optional_args)
            completed = subprocess.run(cmd, shell=False, capture_output=True, text=True)
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr)
            count += 1
f'Finished job "{label}"!'
```
