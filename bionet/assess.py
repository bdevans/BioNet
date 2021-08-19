import os
import gc
import time
import csv

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from bionet.config import (classes,
                           workers, max_queue_size, use_multiprocessing,
                           contrast_level, image_size, interpolation_names,
                           perturbation_columns
                          )
from bionet.preparation import stochastic_perturbations, get_noise_preprocessor


def test_noise_perturbations(model, sim, noise_types, sim_results_dir="", test_set="",
                             test_images_path="", x_test=None, y_test=None):

    seed = sim["seed"]
    train = sim["train"]
    batch = sim["batch"]
    mean = sim["image_mean"]
    std = sim["image_std"]
    colour = sim["colour"]
    save_predictions = sim["save_predictions"]
    image_out_dir = sim["image_out_dir"]
    interpolation_name = interpolation_names[sim["interpolation"]]
    model_name = f'{sim["model"]}_{sim["trial"]}'
    bg_grey = mean / 255  # Background grey level


    if not test_images_path:
        if not test_set:
            test_set = sim["data_set"]  # This is the training set
        load_images_from_disk = False
        assert (x_test is not None) and (y_test is not None)
    else:
        assert test_set
        load_images_from_disk = True

    if (mean is not None) and (std is not None):
        featurewise_normalisation = True
        samplewise_normalisation = False
    else:
        featurewise_normalisation = False
        samplewise_normalisation = True

#     if image_out_dir:
#         save_images = True
#     else:
#         save_images = False

    # Obtain test set labels
    if load_images_from_disk:
        test_batches = ImageDataGenerator().flow_from_directory(
            test_images_path,
            batch_size=batch,
            shuffle=False,
            follow_links=True,
           )
        # test_batches.class_indices.keys()
        outputs = []
        for b in range(len(test_batches)):
            outputs.append(np.argmax(test_batches[b][1], axis=1))
        y_labels = np.squeeze(np.concatenate(outputs))
    else:
        y_labels = np.squeeze(np.argmax(y_test, axis=1))

    # Write results file header
    results_file = os.path.join(sim_results_dir, "metrics", f"{model_name}_perturb_{test_set.lower()}_s{seed}.csv")
    with open(results_file, 'w') as results:
        writer = csv.DictWriter(results, fieldnames=perturbation_columns)
        writer.writeheader()

    # TODO: Optionally test (and generate through the ImageDataGenerator) unperturbed images (L0)

    # Loop over types of noise
    for noise, noise_function, levels in noise_types:
        print(f"[{model_name}] Perturbing test images with {noise} noise...")
        print("-" * 80)

        if noise in stochastic_perturbations:
            # Set the number of workers to 1 for reproducility as this avoids 
            # ordering effects when getting batches with stochastic perturbations
            perturbation_workers = 1
        else:
            perturbation_workers = workers

        for l_ind, level in enumerate(levels):
            print(f"[{l_ind+1:02d}/{len(levels):02d}] level={float(level):6.2f}: ", end='', flush=True)

            t0 = time.time()

            rng = np.random.RandomState(seed=seed+l_ind)  # Ensure a new RNG state for each level
            prep_image = get_noise_preprocessor(noise, noise_function, level, 
                                                contrast_level=contrast_level, 
                                                bg_grey=bg_grey, rng=rng)


            # TODO: Check this is still deterministic when parallelised
            data_gen = ImageDataGenerator(
                preprocessing_function=prep_image,
                featurewise_center=featurewise_normalisation, 
                featurewise_std_normalization=featurewise_normalisation,
                samplewise_center=samplewise_normalisation,
                samplewise_std_normalization=samplewise_normalisation,
                # dtype='float16')
            )

            if featurewise_normalisation:
                # data_gen.fit(x_train)  # Set mean and std
                data_gen.mean = mean
                data_gen.std = std

            if image_out_dir:  # save_images:
                perturbation_image_out_dir = os.path.join(image_out_dir, test_set, noise.replace(' ', '_').lower())
                os.makedirs(perturbation_image_out_dir, exist_ok=True)
                image_prefix = f"L{l_ind:02d}"
            else:
                perturbation_image_out_dir = None
                image_prefix = ""

            if load_images_from_disk:
                gen_test = data_gen.flow_from_directory(
                    test_images_path,  # os.path.join(image_path, 'test'),
                    target_size=image_size,
                    color_mode=colour,  # Not needed?
                    interpolation=interpolation_name,  # Not needed?
                    batch_size=batch,
                    shuffle=False,
                    seed=seed,
                    save_to_dir=perturbation_image_out_dir,
                    save_prefix=image_prefix,
                    follow_links=True,
    #                 subset=None,
    #             classes=classes,
    #             class_mode='categorical',
                )
            else:
                gen_test = data_gen.flow(
                    x_test, 
                    y=y_test, 
                    batch_size=batch,
                    shuffle=False, 
                    seed=seed,
                    save_to_dir=perturbation_image_out_dir,
                    save_prefix=image_prefix
                )

            # Evaluate model performance
            metrics = model.evaluate(gen_test, 
                                     steps=len(gen_test),
                                     verbose=0,
                                     max_queue_size=max_queue_size,
                                     workers=perturbation_workers,
                                     use_multiprocessing=use_multiprocessing
                                    )

            t_elapsed = time.time() - t0

            if train:
                metrics_dict = {metric: score for metric, score in zip(model.metrics_names, metrics)}
                print(f"{metrics_dict} [{t_elapsed:.3f}s]")
            else:
                print(f"{metrics} [{t_elapsed:.3f}s]")


            if save_predictions:

                # NOTE: The results from randomised perturbations do not match 
                # those calculated from the predictions because .evaluate and
                # .predict appear to use generators differently
                # Precise values for Uniform, Salt & Pepper and Phase scrambling
                # are unreproducible. This is likely due to using multiple workers
                # with the ImageDataGenerator. 

                # TODO: Alternatively, generate the same mask for each image
                rng = np.random.RandomState(seed=seed+l_ind)
                prep_image = get_noise_preprocessor(noise, noise_function, level,
                                                    contrast_level=contrast_level,
                                                    bg_grey=bg_grey, rng=rng)

                data_gen = ImageDataGenerator(
                    preprocessing_function=prep_image,
                    featurewise_center=featurewise_normalisation, 
                    featurewise_std_normalization=featurewise_normalisation,
                    samplewise_center=samplewise_normalisation,
                    samplewise_std_normalization=samplewise_normalisation,
                    # dtype='float16'
                )
                if featurewise_normalisation:
                    data_gen.mean = mean
                    data_gen.std = std

                if load_images_from_disk:
                    gen_test = data_gen.flow_from_directory(
                        test_images_path,  # os.path.join(image_path, 'test'),
                        target_size=image_size,
                        color_mode=colour,  # Not needed?
                        interpolation=interpolation_name,  # Not needed?
                        batch_size=batch,
                        shuffle=False,
                        seed=seed,
                        save_to_dir=None,
                        follow_links=True,
                    )
                else:
                    gen_test = data_gen.flow(x_test, y=y_test, batch_size=batch,
                                             shuffle=False, seed=seed, save_to_dir=None)

                predictions = model.predict(gen_test, 
                                            steps=len(gen_test),
                                            verbose=0,
                                            max_queue_size=max_queue_size,
                                            workers=perturbation_workers,
                                            use_multiprocessing=use_multiprocessing)

                predictions_file = os.path.join(sim_results_dir, 'predictions', 
                                                f'{model_name}_perturb_{test_set.lower()}_s{seed}' \
                                                f'_{noise.replace(" ", "_").lower()}_L{l_ind:02d}.csv')

                classifications = np.argmax(predictions, axis=1)

                # Check accuracy based on probabilities matches accuracy from .evaluate
                assert len(classifications) == len(y_labels)
                accuracy = sum(classifications == y_labels) / len(y_labels)
                assert np.isclose(accuracy, metrics[1], atol=1e-6), \
        f"{noise} [{l_ind+1:02d}/{len(levels):02d}]: Calculated: {accuracy} =/= Evaluated: {metrics[1]}"

                # Put predictions into DataFrame
                df_noise = pd.DataFrame(predictions, columns=classes)
                df_noise["Predicted"] = classifications
                df_noise["Class"] = y_labels
                df_noise["Correct"] = classifications == y_labels
                df_noise["Image"] = range(len(y_labels))
                df_noise["Noise"] = [noise] * len(y_labels)
                df_noise["LI"] = [l_ind] * len(y_labels)
                df_noise["Level"] = [level] * len(y_labels)

                df_noise.to_csv(predictions_file, index=False)

                del predictions

            if save_predictions:
                acc = accuracy  # Manual calculation is more accurate, probably due to rounding errors
            else:
                acc = metrics[1]

            # perturbation_columns = ['Model', 'Convolution', 'Base', 'Weights', 'Trial', 'Seed',
            #                         'Set', 'Noise', 'LI', 'Level', 'Loss', 'Accuracy']

            row = {'Model': sim["model"], 'Convolution': sim["convolution"],
                   'Base': sim["base"], 'Weights': str(sim["weights"]),
                   'Trial': sim["trial"], 'Seed': seed,
                   'Set': test_set, 'Noise': noise, 'LI': l_ind, 'Level': level,
                   'Loss': metrics[0], 'Accuracy': acc}
            with open(results_file, 'a') as results:
                writer = csv.DictWriter(results, fieldnames=perturbation_columns)
                writer.writerow(row)

            # Clean up after every perturbation level
            del prep_image
            del gen_test
            del data_gen
            gc.collect()

        print("-" * 80)
