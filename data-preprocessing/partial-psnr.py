import os
import random
import numpy as np
import tensorflow as tf
from IPython.display import Image
from matplotlib.pyplot import imshow

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

subset_samples = np.genfromtxt(path_scores_labels_subset,
                             delimiter=",",
                             dtype=str,
                             encoding='utf-8',
                             usecols=(0)).tolist()

subset_scores = np.genfromtxt(path_scores_labels_subset,
                             delimiter=",",
                             usecols=(1))


def sample_random_image_and_score():
    index = random.randint(0, len(referencepairs))
    sample = referencepairs[index,0]
    reference = referencepairs[index,1]
    sample = tf.io.decode_png(tf.io.read_file(sample))
    reference = tf.io.decode_png(tf.io.read_file(reference))
    score = subset_scores[subset_samples.index("/".join(referencepairs[index,0].split("/")[8:]))] #please fix me!
    return sample,reference,score

def random_patch_from_pair(sample_frame, reference_frame, height=224, width=224):
    assert(sample_frame.shape == reference_frame.shape)
    offset_width = random.randint(0, reference_frame.shape[1]-width)
    offset_height = random.randint(0, reference_frame.shape[0]-height)
    return [tf.image.crop_to_bounding_box(sample_frame,offset_height,offset_width,height,width),
            tf.image.crop_to_bounding_box(reference_frame,offset_height,offset_width,height,width)]


def get_patch_weight(sample, reference, sample_patch, reference_patch):
    partial_psnr = tf.image.psnr(sample_patch, reference_patch, max_val=255).numpy()
    psnr = tf.image.psnr(sample, reference, max_val=255).numpy()
    return partial_psnr / psnr

def get_patch_quality(sample, reference, sample_patch, reference_patch, score):
    return score * get_patch_weight(sample, reference, sample_patch, reference_patch)


sample, reference,score = sample_random_image_and_score()
sample_patch, reference_patch = random_patch_from_pair(sample, reference)

# Display patches and difference
imshow(np.concatenate([sample_patch, reference_patch, reference_patch-sample_patch], axis=1))

get_patch_weight(sample, reference, sample_patch, reference_patch)
get_patch_quality(sample, reference, sample_patch, reference_patch, score)

display(sample, reference, sample_patch, reference_patch)
