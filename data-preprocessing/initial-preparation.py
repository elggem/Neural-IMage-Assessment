import os
import random
import numpy as np

# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment

path_drive_prefix = "../GTTS/"
path_scores_labels = path_drive_prefix + "Labels/ParsedVMAF.csv"
path_scores_labels_subset = path_drive_prefix + "Labels/ParsedVMAF_subset.csv"
path_samples = path_drive_prefix + "Samples/"

path_reference_pair_list = path_drive_prefix + "Labels/referencepairs_nopath.csv"
path_references = path_drive_prefix + "Reference/"

samples = np.genfromtxt(path_scores_labels,
                             delimiter=",",
                             skip_header=1,
                             dtype=str,
                             encoding='utf-8',
                             usecols=(0))

print(samples.shape)

scores = np.genfromtxt(path_scores_labels,
                             delimiter=",",
                             skip_header=1)[:,1:]


# The provided CSV file contains scores from a full dataset of screenshots from
# game transmissions with specific resolution/bitrate combinations. Because
# only a subset of games and images is provided we need to check if files are
# actually present on the disk. Because this operation would be expensive at
# runtime, we generate a list of subset scores here.

def generate_subset_scores():
    outfile = open(path_scores_labels_subset, "w")
    for sample_number, sample in enumerate(samples):
      for score_number, score in enumerate(scores[sample_number]):
        path = "%s%s/%s-%d.png" % (path_samples, sample, sample, score_number)
        if os.path.isfile(path):
          # print("found %d for %s" % (score_number,sample))
          outfile.write("%s/%s-%d.png,%.6f\n" % (sample, sample, score_number, score))
    outfile.close()


# Generate scores for the Reference subset
def generate_reference_list():
    outfile = open(path_reference_pair_list, "w")
    for sample_number, sample in enumerate(samples):
      for score_number, score in enumerate(scores[sample_number]):
        path_sample = "%s%s/%s-%d.png" % (path_samples, sample, sample, score_number)

        reference = "_".join(sample.split("_")[:4])
        path_reference = "%s%s/%s_%04d.png" % (path_references, reference, reference, score_number)
        # print(path_reference)
        if os.path.isfile(path_reference) and os.path.isfile(path_sample):
          # print("found %d for %s" % (score_number,sample))
          outfile.write("%s/%s-%d.png,%s/%s_%04d.png\n" % (sample, sample, score_number, reference, reference, score_number))
    outfile.close()

# generate_reference_list()

referencepairs = np.genfromtxt(path_reference_pair_list,
                             delimiter=",",
                             dtype=str,
                             encoding='utf-8')


# Loads the newly created file.
subset_samples = np.genfromtxt(path_scores_labels_subset,
                             delimiter=",",
                             dtype=str,
                             encoding='utf-8',
                             usecols=(0))
subset_scores = np.genfromtxt(path_scores_labels_subset,
                             delimiter=",",
                             usecols=(1))


# display a random sample and score
from IPython.display import Image

def display_random_image():
    index = random.randint(0, len(subset_samples))
    path = "%s%s" % (path_samples_scaled, subset_samples[index])
    score = subset_scores[index]

    print(score)
    Image(filename=path)


# Resizing function
from PIL import Image
def resize_and_save_samples(image_size=256)
    path_samples_scaled = "%sSamples%d/" % (path_drive_prefix, image_size)

    for sample in subset_samples:
      imagepath_fullsize = "%s%s" % (path_samples, sample)
      imagepath_scaled = "%s%s" % (path_samples_scaled, sample)
      folderpath_scaled = "%s%s" % (path_samples_scaled, sample.split("/")[0])

      if not os.path.isfile(imagepath_scaled):
        try:
          print(imagepath_scaled)
          os.makedirs(folderpath_scaled, exist_ok=True)
          image = Image.open(imagepath_fullsize)
          scaled_image = image.resize((image_size, image_size))
          scaled_image.save(imagepath_scaled)
        except:
          print("file not found")
