from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


def show_samples(dataset):
    # Show a couple images from a pipeline, along with their GT, as a sanity check
    n_cols = 4
    # n_rows = int(ceil(16 / n_cols))
    n_rows = 2
    samples_iter = iter(dataset)
    samples = next(samples_iter)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=109.28)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].imshow(samples['image'][idx])
            x_label = samples['label'][idx].numpy()
            y_label = ''
            idx += 1
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_label)
    # plt.show()
    plt.draw()
    plt.pause(.01)


def pre_process_RPS(ds):
    return ds


def make_RPS_dataset(ds, val_size, train_batch_size, val_batch_size, seed=None):
    val_ds = ds.take(val_size)
    val_ds = pre_process_RPS(val_ds)
    val_ds = val_ds.batch(batch_size=val_batch_size, drop_remainder=False)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = ds.skip(val_size)
    train_ds = train_ds.shuffle(buffer_size=ds.cardinality() - val_size, seed=seed, reshuffle_each_iteration=True)
    train_ds = pre_process_RPS(train_ds)
    train_ds = train_ds.batch(batch_size=train_batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def main():
    seed = 42
    val_p = .2
    data_dir = '/mnt/storage/datasets'
    preprocessed_dir = data_dir + '/rock_paper_scissors/preprocessed'

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

    train_metadata = compile_metadata(dev_ds, preprocessed_dir + '/dev')
    test_metadata = compile_metadata(test_ds, preprocessed_dir + '/test')

    def save_image(sample, filepath):
        image = sample['image']
        image = tf.io.encode_png(image)
        tf.io.write_file(filename=filepath[0], contents=image)
        return sample, filepath

    def preprocess_dataset(dataset, filepaths):
        filepaths_ds = tf.data.Dataset.from_tensor_slices((filepaths,))
        ds = tf.data.Dataset.zip((dataset, filepaths_ds))
        ds = ds.map(save_image, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        ds_iter = iter(ds)
        count = 0
        for count, _ in enumerate(ds_iter):
            pass
        return count

    prepr_dev = preprocess_dataset(dataset=dev_ds, filepaths=train_metadata['x'].to_numpy())
    print(f'Saved {prepr_dev} pre-processed dev. images in {preprocessed_dir}')
    prepr_test = preprocess_dataset(dataset=test_ds, filepaths=test_metadata['x'].to_numpy())
    print(f'Saved {prepr_test} pre-processed test images {preprocessed_dir}')
    print('Count of samples per class in dev. set:')
    print(train_metadata['y'].value_counts().sort_index())
    print('Count of samples per class in test set:')
    print(test_metadata['y'].value_counts().sort_index())

    val_ds_size = int(dev_ds_size * val_p)
    train_ds_size = dev_ds_size - val_ds_size
    pass
    """train_ds, val_ds = make_RPS_dataset(dev_ds,
                                        val_size=val_ds_size,
                                        train_batch_size=16,
                                        val_batch_size=32,
                                        seed=seed)

    count_train_ds = 0
    for item in iter(train_ds):
        count_train_ds += len(item['label'])
    assert count_train_ds == train_ds_size

    count_val_ds = 0
    for item in iter(val_ds):
        count_val_ds += len(item['label'])
    assert count_val_ds == val_ds_size

    assert count_train_ds + count_val_ds == dev_ds.cardinality()

    for ds in (train_ds, val_ds):
        y = []
    for batch in ds:
        y.extend(list(batch['label'].numpy()))
    print(Counter(y))

    labels = {0: 'rock', 1: 'paper', 2: 'scissors'}

    show_samples(train_ds)
    show_samples(val_ds)
    input()"""


if __name__ == '__main__':
    main()
