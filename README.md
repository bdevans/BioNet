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
├── blah.py
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
