# Adapted from https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
# In turn adapted from "Deep Learning with Python", Fracois Chollet, 2017.

import tensorflow as tf
from tensorflow import keras

import matplotlib.cm as cm
import numpy as np


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Returns a Grad-CAM heatmap for a given image and model.
    :param img_array: the image, a batch of tensors of size 1 (containing only one image). The image must have already
    been pre-processed to be fed into the model, in it Input() layer.
    :param model: the Keras model.
    :param last_conv_layer_name: the name of the last convolutional layer of the model, a string, from where Grad-CAM
    shall extract a heatmap.
    :param pred_index: the class for which the heatmap has to be computed, and integer. If set to None, then the heatmap
    is computed for the class predicted for the given image by the model.
    :return: the heatmap, as a numpy array.
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(image, heatmap, cam_path, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap onto an image, and saves the result.
    :param img_path: path to the given image, a string.
    :param heatmap: the heatmap, a numpy array.
    :param cam_path: the path for the file where the image with overlay has to be saved.
    :param alpha: blending between the image and the heatmap, a float between 0 and 1.
    """
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    image = np.uint8(255 * image)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
