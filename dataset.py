from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Input, Model, preprocessing, layers, datasets
from tensorflow.keras.datasets import cifar10

# import tensorflow_io as tfio
# import tensorflow_datasets as tfds
import os
import sys
import numpy as np
from config import xception_cfg as cfg


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope('distort_color') as scope:
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope('distorted_bounding_box_crop') as scope:
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox,
                         fast_mode=True,
                         scope=None):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Additionally it would create image_summaries to display the different
    transformations applied to the image.
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    with tf.name_scope('distort_image') as scope:
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        colors = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox, colors)
        tf.summary.image('image_with_bounding_boxes', image_with_box)

        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox, colors)
        tf.summary.image('images_with_distorted_bounding_box',
                         image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize(x, [height, width]),
            num_cases=num_resize_cases)

        tf.summary.image('cropped_resized_image',
                         tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors. There are 4 ways to do it.
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases=4)

        tf.summary.image('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    """Prepare one image for evaluation.
    If height and width are specified it would output an image with that size by
    applying resize_bilinear.
    If central_fraction is specified it would cropt the central fraction of the
    input image.
    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope('eval_image') as scope:
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize(image, [height, width])
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

def preprocess_image(x, y, is_training=True):
    """Pre-process one image for training or evaluation.
    Args:
      image: 3-D Tensor [height, width, channels] with the image.
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
    Returns:
      3-D float Tensor containing an appropriately scaled image
    Raises:
      ValueError: if user does not provide bounding box
    """
    height = cfg.image_height
    width = cfg.image_width
    bbox = None
    fast_mode = True
    if is_training:
        x = tf.io.read_file(cfg.train_dir + x)
        x = tf.io.decode_jpeg(x, channels=3)
        return preprocess_for_train(x, height, width, bbox, fast_mode), y
    else:
        x = tf.io.read_file(cfg.val_dir + x)
        x = tf.io.decode_jpeg(x, channels=3)
        return preprocess_for_eval(x, height, width), y


def preprocess_cifar(x, y, is_training=True):
    """Pre-process one image for training or evaluation.
    Args:
      image: 3-D Tensor [height, width, channels] with the image.
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
    Returns:
      3-D float Tensor containing an appropriately scaled image
    Raises:
      ValueError: if user does not provide bounding box
    """
    height = cfg.image_height
    width = cfg.image_width
    bbox = None
    fast_mode = True
    if is_training:
        #return preprocess_for_train(x, height, width, bbox, fast_mode), y
        return preprocess_for_eval(x, height, width), y
    else:
        return preprocess_for_eval(x, height, width), y


def base_train_transform(x, y):
    x = tf.io.read_file(cfg.train_dir + x)
    x = tf.io.decode_jpeg(x, channels=3)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize_with_crop_or_pad(image=x, target_height=cfg.image_height, target_width=cfg.image_width)
    x = keras.applications.xception.preprocess_input(x)
    return x, y


def base_val_transform(x, y):
    x = tf.io.read_file(cfg.val_dir + x)
    x = tf.io.decode_jpeg(x, channels=3)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize_with_crop_or_pad(image=x, target_height=cfg.image_height, target_width=cfg.image_width)
    x = keras.applications.xception.preprocess_input(x)
    return x, y


def train_transform(x, y):
    x = tf.io.read_file(cfg.train_dir + x)
    x = tf.io.decode_jpeg(x, channels=3)

    x = tf.cast(x, tf.float32)
    # print(x.get_shape())
    x = tf.image.resize_with_crop_or_pad(image=x, target_height=cfg.image_height, target_width=cfg.image_width)
    x = tf.image.random_brightness(x, max_delta=32. / 255.)
    x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
    x = tf.image.random_hue(x, max_delta=0.2)
    x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
    # x=tf.image.adjust_brightness(image=x, delta=0.4)
    # x=tf.image.adjust_contrast(images=x, contrast_factor=0.4)
    # x=tf.image.adjust_saturation(image=x, saturation_factor=0.4)
    x = tf.image.random_flip_left_right(image=x)
    # x=tfio.experimental.color.rgb_to_bgr(x)
    x = keras.applications.xception.preprocess_input(x)
    # x=tf.transpose(x,[2,0,1])

    return x, y


def val_transform(x, y):
    x = tf.io.read_file(cfg.val_dir + x)
    x = tf.io.decode_jpeg(x, channels=3)

    x = tf.cast(x, tf.float32)
    # print(x.get_shape())
    x = tf.image.resize_with_crop_or_pad(image=x, target_height=cfg.image_height, target_width=cfg.image_width)
    x = tf.image.adjust_brightness(image=x, delta=0.4)
    x = tf.image.adjust_contrast(images=x, contrast_factor=0.4)
    x = tf.image.adjust_saturation(image=x, saturation_factor=0.4)
    x = tf.image.random_flip_left_right(image=x)
    # x=tfio.experimental.color.rgb_to_bgr(x)
    # x=tf.transpose(x,[2,0,1])

    return x, y

# from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# # Example image data, with values in the [0, 255] range
# training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

# cropper = CenterCrop(height=150, width=150)
# scaler = Rescaling(scale=1.0 / 255)

# output_data = scaler(cropper(training_data))
# print("shape:", output_data.shape)
# print("min:", np.min(output_data))
# print("max:", np.max(output_data))


# def limin_preprocess_input(x, data_format=None):
#     """Preprocesses a tensor encoding a batch of images.
#     # Arguments
#         x: input Numpy tensor, 4D.
#         data_format: data format of the image tensor.
#     # Returns
#         Preprocessed tensor.
#     """
#     from keras import backend as K
#     if data_format is None:
#         data_format = K.image_data_format()
#     assert data_format in {'channels_last', 'channels_first'}

#     if data_format == 'channels_first':
#         if x.ndim == 3:
#             # 'RGB'->'BGR'
#             x = x[::-1, ...]
#             # Zero-center by mean pixel
#             x[0, :, :] -= 103.939
#             x[1, :, :] -= 116.779
#             x[2, :, :] -= 123.68
#         else:
#             x = x[:, ::-1, ...]
#             x[:, 0, :, :] -= 103.939
#             x[:, 1, :, :] -= 116.779
#             x[:, 2, :, :] -= 123.68
#     else:
#         # 'RGB'->'BGR'
#         x = x[..., ::-1]
#         # Zero-center by mean pixel
#         x[..., 0] -= 103.939
#         x[..., 1] -= 116.779
#         x[..., 2] -= 123.68

#     x *= 0.017 # scale values

#     return x

# def limin_load_cifar10():
#     # Load the dataset from Keras
#     from keras.datasets import cifar10
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

#     # Preprocessing the dataset
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')

#     #x_train= limin_preprocess_input(x_train)
#     #x_test= limin_preprocess_input(x_test)
#     x_train = x_train.reshape(-1, 299, 299, 3).astype('float32') 
#     x_test = x_test.reshape(-1, 299, 299, 3).astype('float32')
#     y_train = to_categorical(y_train.astype('float32'))
#     y_test = to_categorical(y_test.astype('float32'))
#     print(x_train.type)
#     print(x_test.type)
#     print(y_train.type)
#     print(y_test.type)
#     return (x_train, y_train), (x_test, y_test)

  # from keras.preprocessing.image import ImageDataGenerator
    # train_datagen = ImageDataGenerator(
    #     rescale=1 / 255.0,
    #     rotation_range=20,
    #     zoom_range=0.05,
    #     width_shift_range=0.05,
    #     height_shift_range=0.05,
    #     shear_range=0.05,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     validation_split=0.20)

    # test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    # train_generator = train_datagen.flow_from_directory(
    #     directory=cfg.train_dir,
    #     target_size=(299, 299),
    #     color_mode="rgb",
    #     batch_size=cfg.batch_size,
    #     class_mode="categorical",
    #     subset='training',
    #     shuffle=True,
    #     seed=42
    # )
    # valid_generator = test_datagen.flow_from_directory(
    #     directory=cfg.val_dir,
    #     target_size=(299, 299),
    #     color_mode="rgb",
    #     batch_size=cfg.batch_size,
    #     class_mode="categorical",
    #     subset='validation',
    #     shuffle=True,
    #     seed=42
    # )


def limin_create_cifar10():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(cfg.batch_size)
  val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(cfg.batch_size)
  return train_dataset, val_dataset

def limin_create_dataset():
  print("begin\n")
  train_dataset = keras.preprocessing.image_dataset_from_directory(cfg.train_dir, 
                                                                  batch_size=cfg.batch_size, 
                                                                  image_size=(299, 299))
  # For demonstration, iterate over the batches yielded by the dataset.
  for data, labels in train_dataset:
    print(data.shape)  # (64, 200, 200, 3)
    print(data.dtype)  # float32
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32
  
  #train_dataset = train_dataset.map(lambda x, y: (preprocessing_layer(x), y))
  print("begin val\n")
  val_dataset = keras.preprocessing.image_dataset_from_directory(cfg.val_dir, 
                                                                batch_size=cfg.batch_size, 
                                                                image_size=(299, 299))
  # For demonstration, iterate over the batches yielded by the dataset.
  for data, labels in val_dataset:
    print(data.shape)  # (64, 200, 200, 3)
    print(data.dtype)  # float32
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32

  return train_dataset, val_dataset


def create_dataset():
    assert os.path.exists(cfg.train_file)
    #ndarray
    lists_and_labels = np.loadtxt(cfg.train_file, dtype=str).tolist()
    np.random.shuffle(lists_and_labels)
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    one_shot_labels = keras.utils.to_categorical(labels, cfg.num_classes).astype(dtype=np.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y, True),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    assert os.path.exists(cfg.val_file)
    lists_and_labels = np.loadtxt(cfg.val_file, dtype=str).tolist()
    np.random.shuffle(lists_and_labels)
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    one_shot_labels = keras.utils.to_categorical(labels, cfg.num_classes).astype(dtype=np.int32)
    val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    val_dataset = val_dataset.map(lambda x, y: preprocess_image(x, y, False),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.batch_size)

    return train_dataset, val_dataset


def create_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10).astype(dtype=np.int32)
    y_test = keras.utils.to_categorical(y_test, 10).astype(dtype=np.int32)
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), tf.constant(y_train)))
    train_dataset = train_dataset.map(lambda x, y: preprocess_cifar(x, y, True),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(cfg.batch_size)
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_test), tf.constant(y_test)))
    val_dataset = val_dataset.map(lambda x, y: preprocess_cifar(x, y, False),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.batch_size)
    # val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def create_random_dataset():
    x_train = np.random.randint(0, 256, size=(8192, 299, 299, 3)).astype("float32")
    y_train = np.random.randint(0, 10, size=(8192)).astype("int32")
    x_test = np.random.randint(0, 256, size=(1024, 299, 299, 3)).astype("float32")
    y_test = np.random.randint(0, 10, size=(1024)).astype("int32")

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10).astype(dtype=np.int32)
    y_test = keras.utils.to_categorical(y_test, 10).astype(dtype=np.int32)

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), tf.constant(y_train)))
    # train_dataset = train_dataset.map(lambda x, y: preprocess_cifar(x, y, True),
    #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_test), tf.constant(y_test)))
    # val_dataset = val_dataset.map(lambda x, y: preprocess_cifar(x, y, False),
    #                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset