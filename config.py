import numpy as np
import cv2


# TODO: Replace all definitions of CONST in other source code with:
# from GaborNet.config import CONST

# Models
convolutions_order = ("Original", "Low-pass", "DoG", "Gabor", "Combined")
bases_order = ("ALL-CNN", "VGG-16", "VGG-19")

# Images
data_set = 'CIFAR10'
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)

luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma transform)
_CHANNEL_MEANS = [103.939, 116.779, 123.68]  # BGR

# L = R * 299/1000 + G * 587/1000 + B * 114/1000  # Used by Pillow.Image.convert('L')
_LUMINANCE_MEAN = 123.68 * 0.299 + 116.779 * 0.587 + 103.939 * 0.114  # 117.378639

image_size = (224, 224)
image_shape = image_size + (1,)
interpolation = cv2.INTER_LANCZOS4
colour = 'grayscale'  # 'rgb'
contrast_level = 1  # Proportion of original contrast level for uniform and salt and pepper noise
# mean = 122.61385345458984
# std = 60.87860107421875
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
train_image_stats = {
    cv2.INTER_NEAREST: (122.61930353949222, 60.99213660091195),  # 0: 'nearest'
    cv2.INTER_LANCZOS4: (122.61385345458984, 60.87860107421875)  # 4: 'lanczos'
}

# Project directories
data_dir = '/work/data'
# Output paths
models_dir = '/work/models'
logs_dir = '/work/logs'
results_dir = '/work/results'

image_dir = '/work/data'
all_test_sets = ['line_drawings', 'silhouettes', 'contours']  # , 'scharr']
generalisation_types = ['line_drawings', 'silhouettes', 'contours']
generalisation_sets = ['line_drawings', 'silhouettes', 'contours',
                       'line_drawings_inverted', 'silhouettes_inverted', 'contours_inverted']

# Data format
perturb_columns = ['Model', 'Convolution', 'Base', 'Weights', 'Trial', 'Seed',
                   'Noise', 'Level-Index', 'Level', 'Loss', 'Accuracy']
