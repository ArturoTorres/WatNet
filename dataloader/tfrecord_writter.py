# Conversion of sentinel-2 imagery to .tfrecords
# author: J.A. Torres-Matallana
# Luxembourg Institute of Science and Technology
# since: 2023-06-22; to: 2023-06-22
# source: https://github.com/xinluo2018/WatNet

# With the Dataset, the user can train the WatNet through running the code file
# /trainer.py. Since tfrecords format data is required in the model training,
# the user should convert the .tif format dataset to .tfrecords format dataset by running
# this script firstly.

import os
# os.chdir('..') # uncomment if current dir is this script dir
print(os.getcwd())
import glob
import numpy as np
import tensorflow as tf
from utils.geotif_io import readTiff # first: install conda gdal
from dataloader.path_io import crop_patch
from dataloader.tfrecord_io import image_example

path_tra_tfrecord_scene = 'data/tfrecord-s2/tra_scene.tfrecords'
path_val_tfrecord_patch = 'data/tfrecord-s2/val_patch.tfrecords'
path_val_tfrecord_scene = 'data/tfrecord-s2/val_scene.tfrecords'
tra_scene_paths = sorted(glob.glob('data/dset-s2/tra_scene/*.tif'))
tra_truth_paths = sorted(glob.glob('data/dset-s2/tra_truth/*.tif'))
tra_pair_data = list(zip(tra_scene_paths, tra_truth_paths))
print('tra_data length:', len(tra_pair_data))

val_scene_paths = sorted(glob.glob('data/dset-s2/val_scene/*.tif'))
val_truth_paths = sorted(glob.glob('data/dset-s2/val_truth/*.tif'))
val_pair_data = list(zip(val_scene_paths, val_truth_paths))
print('val_data length:', len(val_pair_data))

# Trainging data: Write to a `.tfrecords` file.
with tf.io.TFRecordWriter(path_tra_tfrecord_scene) as writer:
    for path_scene, path_truth in tra_pair_data:
        print(path_scene)
        scene,_ = readTiff(path_scene)
        truth,_ = readTiff(path_truth)
        scene = np.clip(scene/10000,0,1)
        tf_example = image_example(scene, truth)
        writer.write(tf_example.SerializeToString())

# Val data: Write to a `_scene.tfrecords` file.
with tf.io.TFRecordWriter(path_val_tfrecord_scene) as writer:
    for path_scene, path_truth in val_pair_data:
        print(path_scene)
        scene,_ = readTiff(path_scene)
        truth,_ = readTiff(path_truth)
        scene = np.clip(scene/10000,0,1)
        tf_example = image_example(scene, truth)
        writer.write(tf_example.SerializeToString())

## Validation data: Write to a `_patch.tfrecords` file.
with tf.io.TFRecordWriter(path_val_tfrecord_patch) as writer:
    for i in range(10):    # random croping 10 times for each scene
        for path_scene, path_truth in val_pair_data:
            print(path_scene)
            scene,_ = readTiff(path_scene)
            truth,_ = readTiff(path_truth)
            scene = np.clip(scene/10000,0,1)
            patch, truth = crop_patch(img=scene, truth=truth)
            tf_example = image_example(patch, truth)
            writer.write(tf_example.SerializeToString())