from collections import Counter
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
image_size = (224, 224)
labels = {0: 'rock', 1: 'paper', 2: 'scissors'}
project = 'baseline'
computation_path = 'computation/' + project


def show_samples(dataset, n_rows, n_cols, title):
    # Show a couple images from a pipeline, along with their GT, as a sanity check
    samples_iter = iter(dataset)
    samples = next(samples_iter)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=109.28)
    fig.suptitle(title)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].imshow(samples[0][idx])
            classification = samples[1][idx].numpy()
            x_label = f'{classification} {labels[classification]}'
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


def load_image2(filepath):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_png(image, channels=3)
    return image


def resize_image(sample, y):
    image = sample['image']
    image = tf.image.resize(image, size=image_size)
    image = tf.math.round(image)
    image = tf.cast(image, dtype=tf.uint8)
    sample['image'] = image
    return sample, y


def make_pipeline(filepaths, y, shuffle, batch_size, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), reshuffle_each_iteration=True, seed=seed)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    if not shuffle:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def main():
    train_batch_size = 16
    test_batch_size = 32
    seed = 42
    val_p = .2
    data_dir = '/mnt/storage/datasets'
    preprocessed_dir = data_dir + '/rock_paper_scissors/preprocessed'
    mispredicted_dir = data_dir + '/rock_paper_scissors/mispredicted'
    if not Path(mispredicted_dir).is_dir():
        Path(mispredicted_dir).mkdir(parents=True)

    dev_ds, dev_ds_info = tfds.load('rock_paper_scissors',
                                    split='train',
                                    batch_size=None,
                                    shuffle_files=False,
                                    data_dir=data_dir,
                                    with_info=True)

    test_ds, test_ds_info = tfds.load('rock_paper_scissors',
                                      split='test',
                                      batch_size=None,
                                      shuffle_files=False,
                                      data_dir=data_dir,
                                      with_info=True)

    dev_ds_size = dev_ds_info.splits['train'].num_examples
    test_ds_size = dev_ds_info.splits['test'].num_examples

    def compile_metadata(ds, stem):
        y = []
        ds_iter = iter(ds)
        for batch in ds_iter:
            y.append(batch['label'].numpy())
        metadata = pd.DataFrame({'y': y})
        metadata['x'] = metadata.index.map(lambda i: stem + '{:04d}.png'.format(i))
        return metadata

    dev_metadata = compile_metadata(dev_ds, preprocessed_dir + '/dev')
    test_metadata = compile_metadata(test_ds, preprocessed_dir + '/test')
    n_classes = len(dev_metadata['y'].unique())
    assert len(test_metadata['y'].unique()) == n_classes
    print(f'Samples in the dataset belong to one of {n_classes} classes')

    def save_image(sample, filepath):
        image = sample['image']
        image = tf.io.encode_png(image)
        tf.io.write_file(filename=filepath[0], contents=image)
        return sample, filepath

    def preprocess_dataset(dataset, filepaths):
        filepaths_ds = tf.data.Dataset.from_tensor_slices((filepaths,))
        ds = tf.data.Dataset.zip((dataset, filepaths_ds))
        ds = ds.map(resize_image, num_parallel_calls=AUTOTUNE)
        ds = ds.map(save_image, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        ds_iter = iter(ds)
        count = -1
        for count, _ in enumerate(ds_iter):
            pass
        return count + 1

    if not np.all([Path(filepath).is_file() for filepath in dev_metadata['x']]):
        prepr_dev = preprocess_dataset(dataset=dev_ds, filepaths=dev_metadata['x'].to_numpy())
        print(f'Saved {prepr_dev} pre-processed dev. images in {preprocessed_dir}')
    if not np.all([Path(filepath).is_file() for filepath in test_metadata['x']]):
        prepr_test = preprocess_dataset(dataset=test_ds, filepaths=test_metadata['x'].to_numpy())
        print(f'Saved {prepr_test} pre-processed test images {preprocessed_dir}')

    print('\nCount of samples per class in dev. set:')
    print(dev_metadata['y'].value_counts().sort_index())
    print('Count of samples per class in test set:')
    print(test_metadata['y'].value_counts().sort_index())

    val_ds_size = int(dev_ds_size * val_p)
    train_ds_size = dev_ds_size - val_ds_size
    train_metadata, val_metadata = train_test_split(dev_metadata,
                                                    test_size=val_ds_size,
                                                    random_state=seed,
                                                    shuffle=True,
                                                    stratify=dev_metadata['y'])
    assert len(train_metadata) == train_ds_size
    print('Count of samples per class in train set:')
    print(train_metadata['y'].value_counts().sort_index())
    print('Count of samples per class in val set:')
    print(val_metadata['y'].value_counts().sort_index())

    del dev_ds, dev_ds_info, test_ds, test_ds_info  # Not needed anymore, can free memory

    train_ds = make_pipeline(filepaths=train_metadata['x'],
                             y=train_metadata['y'],
                             shuffle=True,
                             batch_size=train_batch_size,
                             seed=seed)

    val_ds = make_pipeline(filepaths=val_metadata['x'],
                           y=val_metadata['y'],
                           shuffle=False,
                           batch_size=test_batch_size)

    test_ds = make_pipeline(filepaths=test_metadata['x'],
                            y=test_metadata['y'],
                            shuffle=False,
                            batch_size=test_batch_size)

    """
    show_samples(train_ds, 4, 4, title='From the training set')
    show_samples(val_ds, 4, 8, title='From the validation set')
    """

    # Another (slower) way to compute mean and variance across the images, commented out
    """
    images = np.zeros(shape=(len(train_metadata), image_size[0], image_size[1], 3), dtype=float)
    for i, file_name in enumerate(train_metadata['x']):
        images[i] = plt.imread(file_name)
    mean_by_pixel = np.mean(images, axis=(0,1,2))
    var_by_pixel = np.var(images, axis=(0, 1, 2))
    del images
    """

    def compute_images_stats(filepaths):
        stats_pipeline = tf.data.Dataset.from_tensor_slices((filepaths,))
        stats_pipeline = stats_pipeline.map(load_image2, num_parallel_calls=tf.data.AUTOTUNE)
        stats_pipeline = stats_pipeline.map(lambda image: tf.cast(image, dtype=tf.float32), num_parallel_calls=AUTOTUNE)
        stats_pipeline = stats_pipeline.batch(batch_size=len(train_metadata))
        stats_pipeline = stats_pipeline.map(lambda image: image / 255.)
        stats_pipeline = stats_pipeline.prefetch(buffer_size=AUTOTUNE)
        big_batch = next(iter(stats_pipeline))
        assert len(big_batch) == train_ds_size
        mean_by_pixel = tf.math.reduce_mean(big_batch, axis=(0, 1, 2))
        var_by_pixel = tf.math.reduce_variance(big_batch, axis=(0, 1, 2))
        return mean_by_pixel, var_by_pixel

    mean_by_pixel, var_by_pixel = compute_images_stats(train_metadata['x'])

    print(f'Computed mean {mean_by_pixel.numpy()} and variance {var_by_pixel.numpy()} for the training dataset.')

    def make_model_EfficientNetB0(hp, n_classes, dataset_mean, dataset_var, bias_init):

        # The NN expects image pixels to be encoded in the range [0, 255]
        base_model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                          weights='imagenet',
                                                          pooling=None,
                                                          input_shape=image_size + (3,),
                                                          classes=n_classes)

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model.trainable = False
        assert base_model.layers[2].name == 'normalization'
        base_model.layers[2].mean.assign(dataset_mean)
        base_model.layers[2].variance.assign(dataset_var)
        inputs = tf.keras.Input(shape=image_size + (3,))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        bias_initializer = tf.keras.initializers.Constant(bias_init) if bias_init is not None else None
        outputs = tf.keras.layers.Dense(n_classes, bias_initializer=bias_initializer)(x)
        model = tf.keras.Model(inputs, outputs)
        if n_classes > 2:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            selection_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            selection_metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4),
                      loss=loss,
                      metrics=[selection_metric])
        return model

    classes_freq = np.array(train_metadata['y'].value_counts().sort_index(), dtype=float) / len(train_metadata)
    bias_init = np.log(classes_freq / (1 - classes_freq))
    model = make_model_EfficientNetB0(hp=None,
                                      n_classes=3,
                                      dataset_mean=mean_by_pixel,
                                      dataset_var=var_by_pixel,
                                      bias_init=bias_init)
    model.summary()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=computation_path + '/logs')
    history = model.fit(x=train_ds,
                        validation_data=val_ds,
                        epochs=10,
                        verbose=1,
                        callbacks=[tensorboard_cb],
                        shuffle=False)

    test_results = model.evaluate(x=test_ds,
                                  verbose=1,
                                  return_dict=True)
    print(test_results)

    inference_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    test_predict_proba = inference_model.predict(test_ds)
    test_metadata['prediction'] = np.argmax(test_predict_proba, axis=1)

    def duplicate_if_mispredicted(item):
        if item['y'] != item['prediction']:
            destination = f"{mispredicted_dir}/{Path(item['x']).name}"
            Path(item['x']).link_to(destination)

    test_metadata.apply(duplicate_if_mispredicted, axis=1)
    pass


if __name__ == '__main__':
    main()

"""TODO:

switch to different color space
Take validation from test
call adapt()
should I encode the pixels of the dataset as int or float?

"""