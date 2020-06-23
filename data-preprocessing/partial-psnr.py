import os
import random
import numpy as np
import tensorflow as tf
from IPython.display import Image

path_drive_prefix = "/Users/ralf/Documents/github/_TUBForks/GTTS/"
path_scores_labels = path_drive_prefix + "Labels/ParsedVMAF.csv"
path_scores_labels_subset = path_drive_prefix + "Labels/ParsedVMAF_subset.csv"
path_samples = path_drive_prefix + "Samples/"

path_reference_pair_list = path_drive_prefix + "Labels/referencepairs.csv"
path_references = path_drive_prefix + "Reference/"

referencepairs = np.genfromtxt(path_reference_pair_list,
                             delimiter=",",
                             dtype=str,
                             encoding='utf-8')



def sample_random_image_and_score():
    index = random.randint(0, len(referencepairs))
    sample = referencepairs[index,0]
    reference = referencepairs[index,1]
    return sample,reference



sample,reference = sample_random_image_and_score()

Image(filename=sample)
Image(filename=reference)

# Read images from file.
im1 = tf.io.decode_png(tf.io.read_file(reference))
im2 = tf.io.decode_png(tf.io.read_file(sample))
# Compute PSNR over tf.uint8 Tensors.
psnr1 = tf.image.psnr(im1, im2, max_val=255)

# Compute PSNR over tf.float32 Tensors.
im1 = tf.image.convert_image_dtype(im1, tf.float32)
im2 = tf.image.convert_image_dtype(im2, tf.float32)
psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
# psnr1 and psnr2 both have type tf.float32 and are almost equal.

psnr1.numpy()
psnr2.numpy()




























#
