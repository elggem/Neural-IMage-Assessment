import os
import random
import numpy as np
import tensorflow as tf
# from IPython.display import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment

path_drive_prefix = "../GTTS/"
path_scores_labels = path_drive_prefix + "Labels/ParsedVMAF.csv"
path_scores_labels_subset = path_drive_prefix + "Labels/ParsedVMAF_subset.csv"
path_samples = path_drive_prefix + "Samples/"

path_reference_pair_list = path_drive_prefix + "Labels/referencepairs.csv"
path_references = path_drive_prefix + "Reference/"

path_patch_distribution_labels = path_drive_prefix + "Labels/pq_labels_all.csv"


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


print("VMAF Min: %f, Max: %f" % (subset_scores.min(), subset_scores.max()))

def sample_random_image_and_score():
    index = random.randint(0, len(referencepairs))
    sample = referencepairs[index,0]
    reference = referencepairs[index,1]
    print(sample)
    print(reference)
    return load_image_and_score(sample, reference)


def load_image_and_score(sample, reference):
    sample_img = tf.io.decode_png(tf.io.read_file(path_samples + sample))
    reference_img = tf.io.decode_png(tf.io.read_file(path_references + reference))
    score = subset_scores[subset_samples.index(sample)] #please fix me!
    return sample_img,reference_img,score

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


def strided_patches_from_pair(sample_frame, reference_frame, height=224, width=224, stride_factor=4):
    sample_stack = tf.image.extract_patches(images=[sample_frame],
                           sizes=[1, height, width, 1],
                           strides=[1, height/stride_factor, width/stride_factor, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    reference_stack = tf.image.extract_patches(images=[reference_frame],
                           sizes=[1, height, width, 1],
                           strides=[1, height/stride_factor, width/stride_factor, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    return [tf.reshape(sample_stack, (-1,224,224,3)), tf.reshape(reference_stack, (-1,224,224,3))]


# sample, reference,score = sample_random_image_and_score()
# sample_patches, reference_patches =  strided_patches_from_pair(sample, reference)
# sample_patches.shape
# distribution = get_patch_quality(sample, reference, sample_patches, reference_patches, score)
# plt.gca().set(title="xxx", ylabel='Frequency'); plt.hist(distribution, bins=50)
# i = 100
# imshow(np.concatenate([sample_patches[i], reference_patches[i], reference_patches[i]-sample_patches[i]], axis=1))

def generate_distribution_labels():
    outfile = open(path_patch_distribution_labels, "w")
    outfile.write("#sample-path,mean,std,histogram(100dim)\n")
    for i,X in enumerate(referencepairs):
        sam,ref = X
        print("%d / %d" % (i, len(referencepairs)))
        sample, reference, score = load_image_and_score(sam, ref)
        sample_stack, reference_stack =  strided_patches_from_pair(sample, reference)
        patch_quality_distribution = get_patch_quality(sample, reference, sample_stack, reference_stack, score)

        histogram = np.histogram(patch_quality_distribution, bins=100, range=(0.0,150.0), density=True)[0]
        mean = np.array(patch_quality_distribution).mean()
        std = np.std(np.array(patch_quality_distribution))
        outfile.write("%s,%.8e,%.8e,%s\n" % (sam, mean, std, ",".join(format(x, ".8e") for x in histogram)))
    outfile.close()

generate_distribution_labels()

sample_names = np.genfromtxt(path_patch_distribution_labels,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1), skip_header=1)
std = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2), skip_header=1)
means.max()
std.shape
plt.gca().set(title="Mean Distribution of PQ Labels", ylabel='Frequency'); plt.hist(means, bins=100)
plt.gca().set(title="Sigma Distribution of PQ Labels", ylabel='Frequency'); plt.hist(std, bins=100)
# plt.show()

plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':50})
plt.gca().set(title="Mean vs Sigma Distributions of PQ Labels"); plt.scatter(means,std)

# Examples of images with low Sigma:
for i,deviation in enumerate(std):
    if deviation > 15:
        imshow(tf.io.decode_png(tf.io.read_file(path_samples + sample_names[i])))
        plt.show()
        break


def demo_partial_psnr():
    sample, reference,score = sample_random_image_and_score()
    sample_patch, reference_patch = random_patch_from_pair(sample, reference)

    # Display patches and difference
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    imshow(np.concatenate([sample_patch, reference_patch, reference_patch-sample_patch], axis=1))
    imshow(sample)
    imshow(reference)

    get_patch_weight(sample, reference, sample_patch, reference_patch)
    get_patch_quality(sample, reference, sample_patch, reference_patch, score)


# Experiment: Calculating empirical distributions of patch quality across frame
def demo_distribution_patch_quality():
    patch_quality_distribution = []

    sample, reference,score = sample_random_image_and_score()

    # reference_name = "/Users/ralf/Documents/github/_TUBForks/GTTS/Reference/CSGO_30fps_30sec_Part2/CSGO_30fps_30sec_Part2_0001.png"
    reference_name = "/Users/ralf/Documents/github/_TUBForks/GTTS/Reference/WoW_30fps_30sec_Part2/WoW_30fps_30sec_Part2_0001.png"

    for sam,ref in referencepairs:
        if ref == reference_name:
            sample, reference, score = load_image_and_score(sam, ref)
            patch_quality_distribution = []
            for _ in range(1500):
                sample_patch, reference_patch = random_patch_from_pair(sample, reference)
                patch_quality_distribution.append(get_patch_quality(sample, reference, sample_patch, reference_patch, score))

            mean = np.array(patch_quality_distribution).mean()
            sigma = np.std(np.array(patch_quality_distribution))

            plt.gca().set(title="%s\n frame VMAF: %f, mean: %f, sigma: %f" % (sam.split("/")[-1], score, mean, sigma), ylabel='Frequency'); plt.xlim(10,90); plt.hist(patch_quality_distribution, bins=250)
            plt.savefig(sam.split("/")[-1])
            plt.clf()
            print(score)


# Experiment scaling samples and comparing if there is a measurable difference
# RESULTS
# 'nearest': no measurable difference between scales
# 'bilinear', 'bicubic': big difference for 10x10 pixel
def demo_scaling_comparison():
    reference_name = "/Users/ralf/Documents/github/_TUBForks/GTTS/Reference/DiabloIII_30fps_30sec_Part1/DiabloIII_30fps_30sec_Part1_0001.png"

    for sam,ref in referencepairs:
        if ref == reference_name:
            sample, reference, score = load_image_and_score(sam, ref)
            print(sam.split("/")[-2])
            for scales in [[10,10], [256,256], [1080,1080]]:
                sample = tf.image.resize(sample, scales, method='nearest')
                reference = tf.image.resize(reference, scales, method='nearest')
                psnr = tf.image.psnr(sample, reference, max_val=255).numpy()
                print("%d: %f" % (scales[0], psnr))


    imshow(tf.image.resize(sample, [256,256], method='nearest'))
