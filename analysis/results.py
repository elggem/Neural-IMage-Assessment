import os
import random
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


scenario = "pqd"


# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment
path_drive_prefix = "../GTTS/"
path_patch_distribution_labels = path_drive_prefix + "Labels/%s_labels_test.csv" % (scenario)

sample_names = np.genfromtxt(path_patch_distribution_labels,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
gt_means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1))
gt_stds = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2))
gt_pqds = np.genfromtxt(path_patch_distribution_labels,delimiter=",")[:,3:]

path = "./tests_%s/epoch-34_test_pred.csv" % (scenario)
pred_means = np.genfromtxt(path,delimiter=",",usecols=(1))
pred_stds = np.genfromtxt(path,delimiter=",",usecols=(2))
pred_pqds = np.genfromtxt(path,delimiter=",")[:,3:]

# Basic Plots
i = 120
sample_names[i]
plt.plot(gt_pqds[i]/gt_pqds[i].sum()); plt.plot(pred_pqds[i])

gt_pqds[0].sum()
pred_pqds[0].sum()

#VMAF
subset_samples = np.genfromtxt(path_scores_labels_subset,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
path_scores_labels_subset = path_drive_prefix + "Labels/ParsedVMAF_subset.csv"
subset_scores = np.genfromtxt(path_scores_labels_subset,delimiter=",",usecols=(1))
vmaf_scores = []
for sample in sample_names:
    vmaf_score = subset_scores[subset_samples.index(sample)]
    vmaf_scores.append(vmaf_score)
vmaf_scores = np.array(vmaf_scores)
plt.gca().set(title="VMAF distribution of samples", ylabel='Frequency'); plt.hist(vmaf_scores, bins=100); plt.savefig('vmaf_distribution_original_labels.png')

#Resolution
resolution_tags = []
for sample in sample_names:
    for i, resolution in enumerate(['640', '1280', '1920']):
        if resolution in sample:
            resolution_tags.append(i)

# Bitrates
bitrate_tags = []
for sample in sample_names:
    bitrate_tags.append(int(sample.split("_")[-8]))
bitrate_tags = np.array(bitrate_tags)

# Mean Std plots
plt.rcParams.update({'figure.figsize':(5,5), 'figure.dpi':100})
plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(pred_means,pred_stds, c=resolution_tags)

plt.gca().set(title="Mean vs Standard Deviation ground truth"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(gt_means,gt_stds, c=resolution_tags)

plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':70})
plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.xlim(0,120); plt.ylim(0,25); plt.scatter(pred_means,pred_stds, c=resolution_tags, alpha = 0.5, marker='o'); plt.scatter(gt_means,gt_stds, c=resolution_tags, alpha = 0.5, marker='+')


plt.gca().set(title="distribution of standard deviations, red=predicted, blue=gt", ylabel='Frequency'); plt.hist(pred_stds, bins=100, color='red', range=(0,30), alpha = 0.5); plt.hist(gt_stds, bins=100, color='blue', range=(0,30), alpha = 0.5)

plt.gca().set(title="distribution of standard deviations, red=predicted, blue=gt", ylabel='Frequency'); plt.ylim(0,25); plt.hist(pred_means, bins=100, color='red', range=(0,150), alpha = 0.5); plt.hist(gt_means, bins=100, color='blue', range=(0,150), alpha = 0.5)


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
    print("Experiment number %02d, N, %03d, MSE, %.4f" % (i, N, MSE))
