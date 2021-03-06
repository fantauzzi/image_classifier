from matplotlib import pyplot as plt
import re
import datetime
import pickle
from pathlib import Path
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from matplotlib.colors import hsv_to_rgb
import numpy as np
from gradcam import make_gradcam_heatmap, save_gradcam
from time import time

AUTOTUNE = tf.data.AUTOTUNE
image_size = (300, 300)
labels = {}
labels['rock_paper_scissors'] = {0: 'rock', 1: 'paper', 2: 'scissors'}
labels['imagenette'] = {0: 'n01440764',
                        1: 'n02102040',
                        2: 'n02979186',
                        3: 'n03000684',
                        4: 'n03028079',
                        5: 'n03394916',
                        6: 'n03417042',
                        7: 'n03425413',
                        8: 'n03445777',
                        9: 'n03888257'}
project = 'baseline'
computation_path = 'computations/' + project
if not Path(computation_path).is_dir():
    Path(computation_path).mkdir(parents=True, exist_ok=True)
model_checkpoint = computation_path + '/models/best_base.hdf5'
model_checkpoint_ft = computation_path + '/models/best_ft.hdf5'
seed = 42


def show_samples(dataset, n_rows, n_cols, title, use_hsv=False):
    # Show a couple images from a pipeline, along with their GT, as a sanity check
    samples_iter = iter(dataset)
    samples = next(samples_iter)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=109.28)
    fig.suptitle(title)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            image = samples[0][idx]
            if use_hsv:
                image = hsv_to_rgb(image)
            axs[row, col].imshow(image)
            classification = samples[1][idx].numpy()
            x_label = f'{classification} {labels[dataset][classification]}'
            y_label = ''
            idx += 1
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_label)
    # plt.show()
    plt.draw()
    plt.pause(.01)


def load_image(filepath, y):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_png(image, channels=3)
    return image, y


def resize_and_convert_to_int_image(sample, y):
    image = sample['image']
    image = tf.image.resize(image, size=image_size)
    image = tf.math.round(image)
    image = tf.cast(image, dtype=tf.uint8)
    sample['image'] = image
    return sample, y


def image_to_grayscale(sample, y):
    image = sample['image']
    image = tf.image.rgb_to_grayscale(image)
    image = tf.squeeze(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.stack([image, image, image], axis=2)
    sample['image'] = image
    return sample, y


def convert_to_hsv(image):
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.
    image = tf.image.rgb_to_hsv(image)
    return image


def make_pipeline(filepaths, y, shuffle, batch_size, hsv_color_space=True, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), reshuffle_each_iteration=True, seed=seed)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if hsv_color_space:
        ds = ds.map(lambda image, y: (convert_to_hsv(image), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda image, y: (tf.cast(image, dtype=tf.float32) / 255., y))
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    if not shuffle:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def count_components(tensors):
    """
    Returns the total number of components in a sequence of Tensorflow tensors. E.g. if a tensor has shape (3,2,1),
    its components are 3x2x1=6.
    :param tensors: the given sequence of tensors.
    :return: the number of components summed across all the tensors in the given sequence.
    """
    res = sum([np.prod(variable.shape) for variable in tensors])
    return res


def count_weights(layer_or_model):
    """
    Returns the count of trainable, non-trainable and total weights in a given model or layer. The count also includes
    all nested layers, if any.
    :param layer_or_model: a Keras layer or model.
    :return: a 3-tuple, with respectively the count of trainable, non-trainable and total weights.
    """
    trainables_weights, non_trainable_weights, total_weights = 0, 0, 0
    if hasattr(layer_or_model, 'trainable_weights'):
        trainables_weights += count_components(layer_or_model.trainable_weights)
    if hasattr(layer_or_model, 'non_trainable_weights'):
        non_trainable_weights += count_components(layer_or_model.non_trainable_weights)
    if hasattr(layer_or_model, 'weights'):
        total_weights += count_components(layer_or_model.weights)
    return trainables_weights, non_trainable_weights, total_weights


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=.2, seed=seed),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-.1, .3), fill_mode='nearest', seed=seed),
        tf.keras.layers.experimental.preprocessing.RandomRotation(15 / 360, fill_mode='nearest', seed=seed),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=.1,
                                                                     width_factor=.1,
                                                                     fill_mode='nearest',
                                                                     seed=seed)

    ], name='augmentation'
)


def shuffle_dataframe(df, seed=None):
    df = df.sample(n=len(df), replace=False, random_state=seed)
    return df


def main():
    train_batch_size = 16
    test_batch_size = 32
    # dataset = 'rock_paper_scissors'
    dataset = 'imagenette'
    data_dir = '/mnt/storage/datasets'
    preprocessed_dir = f'{data_dir}/{dataset}/preprocessed'
    mispredicted_dir = f'{data_dir}/{dataset}/mispredicted'
    mispredicted_dir_ft = f'{data_dir}/{dataset}/mispredicted_ft'
    augmented_dir = f'{data_dir}/{dataset}/augmented'
    gradcam_path = f'{data_dir}/{dataset}/gradcam'
    str_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    use_hsv = False
    augmentation_ratio = 0
    splits = {'rock_paper_scissors': ('train', 'test'),
              'imagenette': ('train', 'validation')}

    if not Path(gradcam_path).is_dir():
        Path(gradcam_path).mkdir(parents=True)

    # Empty or make the directories for mis-predicted samples. Create them if they don't exist
    for path in (mispredicted_dir, mispredicted_dir_ft):
        if not Path(path).is_dir():
            Path(path).mkdir(parents=True)
        existing_mispredicted = Path(path).glob('*.png')
        for existing_path in existing_mispredicted:
            existing_path.unlink(missing_ok=False)

    # Pre-process the dataset images and save them in `preprocessed_dir`. Store the related metadata in dataframes.
    orig_train_ds, orig_train_ds_info = tfds.load(dataset,
                                                  split=splits[dataset][0],
                                                  batch_size=None,
                                                  shuffle_files=False,
                                                  data_dir=data_dir,
                                                  with_info=True)

    orig_test_ds, orig_test_ds_info = tfds.load(dataset,
                                                split=splits[dataset][1],
                                                batch_size=None,
                                                shuffle_files=False,
                                                data_dir=data_dir,
                                                with_info=True)

    def compile_metadata(ds, stem):
        y = []
        ds_iter = iter(ds)
        for batch in ds_iter:
            y.append(batch['label'].numpy())
        metadata = pd.DataFrame({'y': y})
        metadata['x'] = metadata.index.map(lambda i: stem + '{:04d}.png'.format(i))
        return metadata

    train_metadata = compile_metadata(orig_train_ds, preprocessed_dir + '/dev')
    test_metadata = compile_metadata(orig_test_ds, preprocessed_dir + '/test')
    n_classes = len(train_metadata['y'].unique())
    assert len(test_metadata['y'].unique()) == n_classes
    print(f'Samples in the dataset belong to one of {n_classes} classes')

    def save_image(sample, filepath):
        image = sample['image']
        image = tf.io.encode_png(image)
        tf.io.write_file(filename=filepath[0], contents=image)
        return sample, filepath

    def preprocess_dataset(dataset, filepaths, convert_to_grayscale=False):
        filepaths_ds = tf.data.Dataset.from_tensor_slices((filepaths,))
        ds = tf.data.Dataset.zip((dataset, filepaths_ds))
        if convert_to_grayscale:
            ds = ds.map(image_to_grayscale, num_parallel_calls=AUTOTUNE)
        ds = ds.map(resize_and_convert_to_int_image, num_parallel_calls=AUTOTUNE)
        ds = ds.map(save_image, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        ds_iter = iter(ds)
        count = -1
        for count, _ in enumerate(ds_iter):
            pass
        return count + 1

    if not np.all([Path(filepath).is_file() for filepath in train_metadata['x']]):
        prepr_train = preprocess_dataset(dataset=orig_train_ds,
                                         filepaths=train_metadata['x'].to_numpy(),
                                         convert_to_grayscale=False)
        print(f'Saved {prepr_train} pre-processed dev. images in {preprocessed_dir}')
    if not np.all([Path(filepath).is_file() for filepath in test_metadata['x']]):
        prepr_test = preprocess_dataset(dataset=orig_test_ds,
                                        filepaths=test_metadata['x'].to_numpy(),
                                        convert_to_grayscale=False)
        print(f'Saved {prepr_test} pre-processed test images {preprocessed_dir}')

    train_ds_size = orig_train_ds_info.splits[splits[dataset][0]].num_examples
    test_ds_size = orig_test_ds_info.splits[splits[dataset][1]].num_examples

    del orig_train_ds, orig_test_ds  # Not needed anymore, can free memory

    print('\nCount of samples per class in train set:')
    print(train_metadata['y'].value_counts().sort_index())
    print('Count of samples per class in original test set:')
    print(test_metadata['y'].value_counts().sort_index())

    def make_augmented_filepath(filepath, i):
        stem = Path(filepath).stem
        res = '{}/{}_{:02d}.png'.format(augmented_dir, stem, i)
        return res

    def load_augment_save(filepath, y, augmented_filepath):
        image, _ = load_image(filepath, y)
        image = tf.expand_dims(image, axis=0)
        augmented = data_augmentation(image)
        augmented = tf.squeeze(augmented, axis=0)
        augmented_png = tf.io.encode_png(augmented)
        tf.io.write_file(augmented_filepath, augmented_png)
        return filepath, y, augmented_filepath

    def augment(metadata, ratio):
        if ratio == 0:
            return metadata
        augmented_all = None  # Just to avoid bogus warning from IntelliJ
        for i in range(ratio):
            augmented = metadata.copy()
            augmented['augmented'] = augmented['x'].map(lambda filepath: make_augmented_filepath(filepath, i))
            augmented_all = augmented if i == 0 else pd.concat([augmented_all, augmented], ignore_index=True)

        if not np.all([Path(filepath).is_file() for filepath in augmented_all['augmented']]):
            ds = tf.data.Dataset.from_tensor_slices(
                (augmented_all['x'], augmented_all['y'], augmented_all['augmented']))
            ds = ds.map(load_augment_save, num_parallel_calls=AUTOTUNE)
            ds = ds.batch(batch_size=32, drop_remainder=False)
            ds = ds.prefetch(buffer_size=AUTOTUNE)

            ds_iter = iter(ds)
            for _ in ds_iter:
                pass

        augmented_all.drop(['x'], axis=1, inplace=True)
        augmented_all.rename({'augmented': 'x'}, axis=1, inplace=True)
        augmented_all = pd.concat([metadata, augmented_all], ignore_index=True)
        return augmented_all

    train_metadata = augment(train_metadata, ratio=augmentation_ratio)
    train_metadata = shuffle_dataframe(train_metadata, seed=seed)

    # Split the test dataset in two, making a validation and a test set
    val_ds_size, test_ds_size = test_ds_size // 2, test_ds_size - test_ds_size // 2

    val_metadata, test_metadata = train_test_split(test_metadata,
                                                   test_size=val_ds_size,
                                                   random_state=seed,
                                                   shuffle=True,
                                                   stratify=test_metadata['y'])
    print('Count of samples per class in val set after test-val split:')
    print(val_metadata['y'].value_counts().sort_index())
    print('Count of samples per class in test set after test-val split:')
    print(test_metadata['y'].value_counts().sort_index())

    # Make pipelines for train/val/test
    train_ds = make_pipeline(filepaths=train_metadata['x'],
                             y=train_metadata['y'],
                             shuffle=True,
                             batch_size=train_batch_size,
                             hsv_color_space=use_hsv,
                             seed=seed)

    val_ds = make_pipeline(filepaths=val_metadata['x'],
                           y=val_metadata['y'],
                           shuffle=False,
                           hsv_color_space=use_hsv,
                           batch_size=test_batch_size)

    test_ds = make_pipeline(filepaths=test_metadata['x'],
                            y=test_metadata['y'],
                            shuffle=False,
                            hsv_color_space=use_hsv,
                            batch_size=test_batch_size)

    # show_samples(train_ds, 4, 4, title='From the training set', use_hsv=use_hsv)
    # show_samples(val_ds, 4, 8, title='From the validation set', use_hsv=use_hsv)
    # show_samples(test_ds, 4, 8, title='From the test set', use_hsv=use_hsv)

    def compile_model(model, n_classes, learning_rate):
        if n_classes > 2:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            selection_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            selection_metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4),
                      loss=loss,
                      metrics=[selection_metric])

    def make_model_EfficientNet(n_classes, dataset, bias_init, stats_filepath, augment=False):
        base_model = tf.keras.applications.EfficientNetB3(include_top=False,
                                                          weights='imagenet',
                                                          pooling=None,
                                                          input_shape=image_size + (3,),
                                                          classes=n_classes)

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input  # This is the identity function
        base_model.trainable = False

        normalization_layer = base_model.layers[2]
        assert normalization_layer.name == 'normalization'
        if Path(stats_filepath).is_file():
            with open(stats_filepath, 'rb') as pickle_f:
                stats = pickle.load(pickle_f)
                normalization_layer.mean = stats['mean']
                normalization_layer.variance = stats['variance']
            print(f'Loaded dataset stats from file {stats_filepath}')
        else:
            print('Computing dataset stats')
            normalization_layer.adapt(dataset)
            with open(stats_filepath, 'wb') as pickle_f:
                pickle_this = {'mean': normalization_layer.mean,
                               'variance': normalization_layer.variance}
                pickle.dump(pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved dataset stats in file {stats_filepath}')

        assert base_model.layers[1].name == 'rescaling'
        # With this change, the NN expects image pixels to be encoded in the range [0, 1]
        base_model.layers[1].scale = 1.

        inputs = tf.keras.Input(shape=image_size + (3,))
        if augment:
            x = data_augmentation(inputs)
        else:
            x = inputs
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.Activation('relu', name='final_activation')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        bias_initializer = tf.keras.initializers.Constant(bias_init) if bias_init is not None else None
        outputs = tf.keras.layers.Dense(n_classes, bias_initializer=bias_initializer)(x)
        model = tf.keras.Model(inputs, outputs)
        compile_model(model, n_classes=n_classes, learning_rate=1e-5)
        return model

    classes_freq = np.array(train_metadata['y'].value_counts().sort_index(), dtype=float) / len(train_metadata)
    bias_init = np.log(classes_freq / (1 - classes_freq))

    ''' Pipeline enumerating all images (no ground truth), it will be passed to model instantiation to compute
    the per-channel average and variance of the images; these statistics are used to normalize the images.'''
    images_only_ds = train_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
    images_only_ds = images_only_ds.prefetch(buffer_size=AUTOTUNE)

    model = make_model_EfficientNet(n_classes=n_classes,
                                    dataset=images_only_ds,
                                    bias_init=bias_init,
                                    stats_filepath=computation_path + '/model_stats.pickle',
                                    augment=False)
    del images_only_ds  # Not needed anymore, can free memory
    model.summary()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f'{computation_path}/logs/base-{str_time}',
                                                    profile_batch=0,
                                                    histogram_freq=1)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint,
                                                       monitor='val_sparse_categorical_accuracy',
                                                       verbose=1,
                                                       save_best_only=True,
                                                       mode='auto')

    start_time = time()
    history = model.fit(x=train_ds,
                        validation_data=val_ds,
                        epochs=3,
                        verbose=1,
                        callbacks=[tensorboard_cb, checkpoint_cb],
                        shuffle=False)
    duration = round(time()- start_time)
    print(f'Training of base model completed in {duration} seconds.')

    best_model = tf.keras.models.load_model(filepath=model_checkpoint, compile=True)

    def prepare_for_fine_tuning_efficientnetb0(model, n_classes, learning_rate):
        for layer in model.layers:
            if layer.name == 'efficientnetb3':
                base_model = layer
                break
        else:
            assert False
        base_model.trainable = True

        pattern = re.compile('block(\d)a_expand_conv')
        block_ends = []
        block_end_names = []
        for i, layer in enumerate(base_model.layers):
            match = pattern.match(layer.name)
            if match is not None:
                block_ends.append(i)
                block_end_names.append(layer.name)
        last_frozen_layer = 3  # from 0 to 5, included

        for i in range(block_ends[last_frozen_layer]):
            base_model.layers[i].trainable = False

        compile_model(base_model, n_classes=n_classes, learning_rate=learning_rate)

    prepare_for_fine_tuning_efficientnetb0(best_model, n_classes=n_classes, learning_rate=1e-6)
    best_model.summary()

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f'{computation_path}/logs/ft-{str_time}',
                                                    profile_batch=0,
                                                    histogram_freq=1)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_ft,
                                                       monitor='val_sparse_categorical_accuracy',
                                                       verbose=1,
                                                       save_best_only=True,
                                                       mode='auto')

    start_time = time()
    history_ft = best_model.fit(x=train_ds,
                                validation_data=val_ds,
                                epochs=5,
                                verbose=1,
                                callbacks=[tensorboard_cb, checkpoint_cb],
                                shuffle=False)
    duration = round(time()- start_time)
    print(f'Fine-tuning completed in {duration} seconds.')

    best_model_ft = tf.keras.models.load_model(filepath=model_checkpoint_ft, compile=True)

    test_results = best_model_ft.evaluate(x=test_ds,
                                          verbose=1,
                                          return_dict=True)
    print(test_results)

    inference_model = tf.keras.Sequential([best_model_ft, tf.keras.layers.Softmax()])
    test_predict_proba = inference_model.predict(test_ds)
    test_metadata['prediction'] = np.argmax(test_predict_proba, axis=1)

    def duplicate_if_mispredicted(item):
        if item['y'] != item['prediction']:
            destination = f"{mispredicted_dir}/{Path(item['x']).name}"
            Path(item['x']).link_to(destination)

    test_metadata.apply(duplicate_if_mispredicted, axis=1)

    img_array = next(iter(test_ds))[0][0].numpy()
    # Put the image in a batch of size 1, what the model for Gra-CAM expects
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array=img_array,
                                   model=best_model_ft,
                                   last_conv_layer_name='final_activation')

    # gradcam_path = computation_path + '/gradcam'
    progr = 0
    for batch in test_ds:
        for image in batch[0]:
            img_array = image.numpy()
            # Put the image in a batch of size 1, what the model for Gra-CAM expects
            img_array = np.expand_dims(img_array, axis=0)
            heatmap = make_gradcam_heatmap(img_array=img_array,
                                           model=best_model_ft,
                                           last_conv_layer_name='final_activation')
            save_gradcam(image=image, heatmap=heatmap, cam_path='{}/{:04d}.png'.format(gradcam_path, progr))
            progr += 1


if __name__ == '__main__':
    main()

"""TODO:
Parallel feeding to the NN of multiple batches
Try anther dataset, e.g. immaginettes
"""
