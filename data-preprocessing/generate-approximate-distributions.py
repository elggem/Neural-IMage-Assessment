import os
import random
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment
path_drive_prefix = "../GTTS/"
path_patch_distribution_labels = path_drive_prefix + "Labels/pq_labels_all.csv"
path_approximate_normal_variable_sigma_labels = path_drive_prefix + "Labels/anv_labels_all.csv"
path_approximate_normal_fixed_sigma_labels = path_drive_prefix + "Labels/anf_labels_all.csv"

sample_names = np.genfromtxt(path_patch_distribution_labels,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1), skip_header=1)
stds = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2), skip_header=1)
pqds = np.genfromtxt(path_patch_distribution_labels,delimiter=",", skip_header=1)[:,3:]


def generate_normal_distribution(mean, std):
    distribution = []
    for _ in range(1500): # pqd is generated from 1500 samples
        distribution.append(np.random.normal(loc=mean, scale=std))
    return np.histogram(np.array(distribution), bins=100, range=(0.0,150.0), density=True)[0]



outfile_anv = open(path_approximate_normal_variable_sigma_labels, "w")
outfile_anf = open(path_approximate_normal_fixed_sigma_labels, "w")

fixed_sigma = stds.mean()

for i,sam in enumerate(sample_names):
    print("%i/%i" % (i, len(sample_names)))
    mean = means[sample_names.index(sam)-1]
    std  = stds[sample_names.index(sam)-1]
    pqd  = pqds[sample_names.index(sam)-1]
    normal_dist_variable_sigma = generate_normal_distribution(mean, std)
    normal_dist_fixed_sigma = generate_normal_distribution(mean, fixed_sigma)
    outfile_anv.write("%s,%.8e,%.8e,%s\n" % (sam, mean, std, ",".join(format(x, ".8e") for x in normal_dist_variable_sigma)))
    outfile_anf.write("%s,%.8e,%.8e,%s\n" % (sam, mean, fixed_sigma, ",".join(format(x, ".8e") for x in normal_dist_fixed_sigma)))

outfile_anv.close()
outfile_anf.close()
