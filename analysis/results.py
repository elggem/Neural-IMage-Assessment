import os
import random
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment
path_drive_prefix = "../GTTS/"
path_patch_distribution_labels = path_drive_prefix + "Labels/pqd_labels_test.csv"
path_patch_distribution_predictions = "./tests_pqd/epoch-28_test_pred.csv"

sample_names = np.genfromtxt(path_patch_distribution_labels,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
gt_means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1))
gt_stds = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2))
gt_pqds = np.genfromtxt(path_patch_distribution_labels,delimiter=",")[:,3:]


path = "./results/001-not-normed/tests_pqd/epoch-50_test_pred.csv"
pred_means = np.genfromtxt(path,delimiter=",",usecols=(1))
pred_stds = np.genfromtxt(path,delimiter=",",usecols=(2))
pred_pqds = np.genfromtxt(path,delimiter=",")[:,3:]

# Basic Plots
sample_names[100]
plt.plot(gt_pqds[100])
plt.plot(pred_pqds[100])
gt_pqds[0].sum()
pred_pqds[0].sum()

# Mean Std plots
plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':50})
plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(pred_means,pred_stds)

plt.gca().set(title="Mean vs Standard Deviation ground truth"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(gt_means,gt_stds)


# Experimental Results for low/medium/high resolution/bitrate MSE
def get_available_bitrates():
    bitrates = []
    for sample in sample_names:
        bitrates.append(int(sample.split("_")[11]))
    return np.unique(bitrates)

# final result split
category_indices = ["low", "medium", "high"] # 'all'
resolutions = ['640x480', '1280x720', '1920x1080']
bitrates = get_available_bitrates()
split_index = 5; bitrates_splits = [bitrates[:split_index], bitrates[split_index:split_index*2], bitrates[split_index*2:]]

def sample_matches(sample, resolution="all", bitrate="all"):
    match = False
    if resolution=='all' or resolutions[category_indices.index(resolution)] in sample:
        match = True
    if not bitrate=='all':
        sample_bitrate = int(sample.split("_")[11])
        if not sample_bitrate in bitrates_splits[category_indices.index(bitrate)]:
            match = False
    return match


experiments = [['all', 'all'], ['high', 'all'], ['medium', 'all'], ['low', 'all'], ['all', 'high'], ['all', 'medium'], ['all', 'low'], ['high', 'high'], ['medium', 'medium'], ['low', 'low']]

for i, experiment in enumerate(experiments):
    N = 0
    MSE = 0
    for sample in sample_names:
        if sample_matches(sample, resolution=experiment[0], bitrate=experiment[1]):
            N += 1
            MSE += (gt_means[sample_names.index(sample)] - pred_means[sample_names.index(sample)]) ** 2
    MSE /= N
    print("Experiment number %02d, N=%03d, MSE=%.4f" % (i, N, MSE))
