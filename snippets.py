# Another (slower) way to compute mean and variance across the images, commented out
"""
images = np.zeros(shape=(len(train_metadata), image_size[0], image_size[1], 3), dtype=float)
for i, file_name in enumerate(train_metadata['x']):
    images[i] = plt.imread(file_name)
mean_by_pixel = np.mean(images, axis=(0,1,2))
var_by_pixel = np.var(images, axis=(0, 1, 2))
del images
"""

"""def compute_images_stats(filepaths):
    stats_pipeline = tf.data.Dataset.from_tensor_slices((filepaths,))
    stats_pipeline = stats_pipeline.map(load_image2, num_parallel_calls=tf.data.AUTOTUNE)
    stats_pipeline = stats_pipeline.map(lambda image: tf.cast(image, dtype=tf.float32), num_parallel_calls=AUTOTUNE)
    stats_pipeline = stats_pipeline.batch(batch_size=len(train_metadata))
    stats_pipeline = stats_pipeline.map(lambda image: image / 255., num_parallel_calls=AUTOTUNE)
    stats_pipeline = stats_pipeline.prefetch(buffer_size=AUTOTUNE)
    big_batch = next(iter(stats_pipeline))
    assert len(big_batch) == train_ds_size
    mean_by_pixel = tf.math.reduce_mean(big_batch, axis=(0, 1, 2))
    var_by_pixel = tf.math.reduce_variance(big_batch, axis=(0, 1, 2))
    return mean_by_pixel, var_by_pixel

mean_by_pixel, var_by_pixel = compute_images_stats(train_metadata['x'])

print(f'Computed mean {mean_by_pixel.numpy()} and variance {var_by_pixel.numpy()} for the training dataset.')"""
