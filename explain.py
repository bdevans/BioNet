import os
import json
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt
from tf_explain.core.grad_cam import GradCAM

from GaborNet.preparation import (get_directory_generator, 
                                  get_noise_preprocessor, 
                                  invert_luminance)
from GaborNet.utils import load_model, find_conv_layer
# from GaborNet.preparation import _CHANNEL_MEANS, _LUMINANCE_MEAN


luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma
interpolation = cv2.INTER_LANCZOS4
mean = 122.61385345458984
std = 60.87860107421875
train_stats = {
    'nearest': (122.61930353949222, 60.99213660091195),
    'lanczos': (122.61385345458984, 60.87860107421875)
}

image_dir = '/work/data'
all_test_sets = ['line_drawings', 'silhouettes', 'contours']  # , 'scharr']


def plot_grad_cam(data_set, model_name, test_set=None, image_weight=0.7, fig_sf=2, save_figure=False):

    rescale = 1/255
    n_classes = 10
    n_samples = 10


    model = load_model(data_set, model_name)

    # Instantiation of the explainer
    explainer = GradCAM()
    
    if test_set is None:
        test_sets = all_test_sets
    else:
        assert test_set in all_test_sets
        test_sets = [test_set]
    
    for test_set in test_sets:
        # root = f'/work/data/{test_set}/'
        root = os.path.join(image_dir, test_set)
        for invert_test_images in [False, True]:
            if invert_test_images:
                prep_image = get_noise_preprocessor("Invert", invert_luminance, level=1, rescale=rescale)
                annotated_test_set = f'{test_set}_inverted'
            else:
                prep_image = get_noise_preprocessor("None", rescale=rescale)
                annotated_test_set = f'{test_set}'

            fig, axes = plt.subplots(nrows=n_classes, ncols=n_samples, 
                                    figsize=(fig_sf*n_samples, fig_sf*n_classes))
            i = 0
            for c_ind, category in enumerate(sorted(os.listdir(root))):
                for s_ind, img in enumerate(sorted(os.listdir(os.path.join(root, category)))):
                    if annotated_test_set.startswith('scharr') and s_ind >= 10:
                        break
                    full_path = os.path.join(root, category, img)
                    image = plt.imread(full_path) * 255
                    if image.shape == (224, 224):
                        image = image[..., np.newaxis]
                    if image.shape == (224, 224, 3):
                        image = np.dot(image, luminance_weights)[..., np.newaxis]
        #             image = np.squeeze(prep_image(image))
                    image = prep_image(image)
                    
                    image -= mean
                    image /= std

                    data = ([image], None)

                    # Call to explain() method
                    output = explainer.explain(data, model, class_index=c_ind, image_weight=image_weight)

                    # print(np.amin(output), np.amax(output))
                    axes[c_ind, s_ind].imshow(output, vmin=0, vmax=255)  # cmap='gray', 
                    axes[c_ind, s_ind].set_xticks([])
                    axes[c_ind, s_ind].set_yticks([])
                    axes[c_ind, s_ind].get_xaxis().set_visible(False)
                    axes[c_ind, s_ind].get_yaxis().set_visible(False)
                    axes[c_ind, s_ind].set_axis_off()

                    if s_ind == 0:
                        axes[s_ind, c_ind].set_ylabel(category.capitalize())
                    i += 1

            fig.subplots_adjust(0.02,0.02,0.98,0.98)
            if save_figure:
                fig_output_dir = f'/work/results/{label}/{model_name}/grad_cam/'
                os.makedirs(fig_output_dir, exist_ok=True)
                output = f'{fig_output_dir}/{test_set}{"_invert" if invert_test_images else ""}.png'
                fig.savefig(output)
            f"Created figure: {output}"


def get_activations(image_path, data_set, model_name, layer_id=None, verbose=0):

    # data_root = '/work/data'

    model = load_model(data_set, model_name, verbose=verbose)
    # Model to examine
    # model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)
    input_shape = model.input.get_shape().as_list()
    if verbose:
        print(f"Input shape: {input_shape}")

    # TODO
    if layer_id is None:
        (layer_name, layer_ind) = find_conv_layer(model)
        layer = model.get_layer(name=layer_id)
        outputs = [layer.output]
    elif isinstance(layer_id, str):
        # Name
        layer_name = layer_id
        layer = model.get_layer(name=layer_id)
        # layer_ind
        outputs = [layer.output]
    elif isinstance(layer_id, int):
        # Index
        layer_ind = layer_id
        layer = model.get_layer(index=layer_id)
        layer_name = layer.name
        outputs = [layer.output]
    elif isinstance(layer_id, (list, tuple)):
        # Assume list of indicies
        outputs = [model.get_layer(index=layer_ind).output 
                   for layer_ind in layer_id]
        layer_name = ", ".join(map(str, layer_id))
    else:
        print(f"Error! Unknown layer type: {layer_id} ({type(layer_id)})")

    print(f"Getting activations of layer {layer_name} from {data_set}/{model_name}...")

# #     layers_name = [layer_name.lower()]
#     # Get the outputs of layers we want to inspect
#     outputs = [
#         layer.output for layer in model.layers
#         if layer.name.lower() == layer_name.lower()
# #         if layer.name.lower() in layers_name
#     ]

    if verbose:
        print("Inputs: ")
        pprint(model.inputs)
        print("Outputs: ")
        pprint(outputs)

    # Create a connection between the input and those target outputs
    activations_model = tf.keras.models.Model(model.inputs, outputs=outputs)
    activations_model.compile(optimizer='adam', loss='categorical_crossentropy')

    if isinstance(image_path, str):
        image_paths = [image_path]
    else:
        image_paths = image_path
        assert isinstance(image_paths, (list, tuple))

    activations = {}
    for image_path in image_paths:
        print(f"{image_path}: ", end='')
        if os.path.isfile(image_path):
            # Image to pass as input
            # TODO: replace with the same routines as used elsewhere
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[1:-1], #(224, 224), 
                                                        color_mode="grayscale", interpolation="lanczos")
            img = tf.keras.preprocessing.image.img_to_array(img)

            # print(np.amin(img), np.amax(img), flush=True)
            # img = np.expand_dims(np.dot(img, luminance_weights), axis=-1)
            # cv2.resize(image, dsize=image_size, interpolation=interpolation)
            img[img < 0] = 0
            img[img > 255] = 255

            img -= mean
            img /= std

            activations[image_path] = activations_model.predict(np.array([img]))
        elif os.path.isdir(image_path):

            gen_test = get_directory_generator(image_path, 
                                            # preprocessing_function=preprocessing_function, 
                                            mean=mean, std=std, # invert=image_path.endswith('invert'), 
                                            image_size=(224, 224), colour='grayscale', 
                                            batch_size=10, shuffle=False,
                                            interpolation='lanczos')
            activations[os.path.basename(image_path)] = activations_model.predict(gen_test)

    # Get their outputs
    return activations


def plot_activations(image_path, data_set, model_name, layer_name=None, fig_sf=2, verbose=0):
    
    activations = get_activations(image_path, data_set, model_name, layer_name)
    
    shape = activations[image_path].shape
    # np.squeeze(activations_1)
    # print(shape)
    nrows = int(np.sqrt(shape[-1]))
    ncols = int(np.ceil(shape[-1]/nrows))
    while nrows * ncols != shape[-1]:
        ncols += 1
        nrows -= 1
    # print(nrows, ncols)

    # if verbose:
    #     plt.imshow(np.squeeze(img), cmap="gray")
    #     plt.gca().set_axis_off()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             sharex=True, sharey=True, squeeze=False,
                             figsize=(ncols*fig_sf, nrows*fig_sf))

    for feat, ax in enumerate(axes.ravel()):
        ax.imshow(activations[image_path][0, :, :, feat])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()

    return (fig, axes)


def plot_occlusion_sensitivity(data_set, model_name, image_path, class_index, fig_sf=2, verbose=0):
    
    # Create function to apply a grey patch on an image
    def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
        patched_image = np.array(image, copy=True)
        patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5

        return patched_image
    
    print(f"Plotting occlusion sensitivity for {data_set}/{model_name} on {image_path}...")
    model = load_model(data_set, model_name, verbose=verbose)
    # Model to examine
    # model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)
    input_shape = model.input.get_shape().as_list()

    # Image to pass as input
    # TODO: replace with the same routines as used elsewhere
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[1:-1], #(224, 224), 
                                                color_mode="grayscale", interpolation="lanczos")
    img = tf.keras.preprocessing.image.img_to_array(img) # * 255

    # print(np.amin(img), np.amax(img), flush=True)
    # img = np.expand_dims(np.dot(img, luminance_weights), axis=-1)
    # cv2.resize(image, dsize=image_size, interpolation=interpolation)
    img[img < 0] = 0
    img[img > 255] = 255

    img -= mean
    img /= std


    # CAT_CLASS_INDEX = 5 #281  # Imagenet tabby cat class index
    PATCH_SIZE = 32 #40

    sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[0], PATCH_SIZE):
        for top_left_y in range(0, img.shape[1], PATCH_SIZE):
            patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)
            predicted_classes = model.predict(np.array([patched_image]))[0]
            confidence = predicted_classes[class_index]

            # Save confidence for this specific patched image in map
            sensitivity_map[
                top_left_y:top_left_y + PATCH_SIZE,
                top_left_x:top_left_x + PATCH_SIZE,
            ] = confidence


    plt.imshow(sensitivity_map)
    plt.colorbar()

    return #(fig, axes)


def plot_most_activating_features(data_set, model_name, layer=None, filter_index=None, 
                                  epochs=100, step_size=1., ax=None, fig_sf=2):

    # stop_layer = 'flatten'
    # stop_layer = 'block3_conv1'

    # Create a connection between the input and the target layer
    model = load_model(data_set, model_name)

    if layer is None:
        layer_name, layer_ind = find_conv_layer(model)
        # layer = model.layers[layer_ind]
        layer = layer_ind
    if isinstance(layer, str):
        layer = model.get_layer(name=layer_name)
    elif isinstance(layer, int):
        layer = model.get_layer(index=layer)

    # stop_index = model.layers.index(model.get_layer(stop_layer))
    # nrows = stop_index-1  # len(model.layers)-1
    # fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(3, 3*nrows))
    # fig = plt.figure()

    if isinstance(filter_index, str) and filter_index == 'all':
        shape = layer.output.shape
        nrows = int(np.sqrt(shape[-1]))
        ncols = int(np.ceil(shape[-1]/nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                sharex=True, sharey=True, squeeze=False,
                                figsize=(ncols*fig_sf, nrows*fig_sf))
        filter_indices = range(int(shape[-1]))
        title_prefix = ""
        plt.suptitle(f"{data_set}/{model_name}")
    elif filter_index is None:
        shape = layer.output.shape
        filter_indices = [np.random.choice(shape[-1])]
        title_prefix = f"{data_set}/{model_name}: "
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        axes = np.array([ax])
    else:
        # nrows = 1
        # ncols = 1
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        axes = np.array([ax])
        filter_indices = [filter_index]
        title_prefix = f"{data_set}/{model_name}: "
    # for ax, layer in zip(axes.ravel(), model.layers[1:stop_index]):
        
    submodel = tf.keras.models.Model([model.inputs[0]], [layer.output])

    for ax, filter_index in zip(axes.ravel(), filter_indices):
        # Initiate random noise
        input_img_data = np.random.random((1, 224, 224, 1))
        input_img_data = (input_img_data - 0.5) * 20 #+ 128.
        # Cast random noise from np.float64 to tf.float32 Variable
        input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))
        
        # Iterate gradient ascents
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                outputs = submodel(input_img_data)
                loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
            grads = tape.gradient(loss_value, input_img_data)
            normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            input_img_data.assign_add(normalized_grads * step_size)
        cbmap = ax.imshow(np.squeeze(input_img_data.numpy()))
        ax.set_title(f"{title_prefix}{layer.name}[{filter_index}]")
        fig.colorbar(cbmap, ax=ax)
    # input_img_data.numpy().shape
    # plt.imshow(np.squeeze(input_img_data.numpy()))
