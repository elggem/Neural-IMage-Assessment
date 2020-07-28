import os
import random
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


scenario = "anf"


# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment
path_drive_prefix = "../GTTS/"
path_patch_distribution_labels = path_drive_prefix + "Labels/%s_labels_test.csv" % (scenario)
path_scores_labels_subset = path_drive_prefix + "Labels/ParsedVMAF_subset.csv"

sample_names = np.genfromtxt(path_patch_distribution_labels,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
gt_means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1))
gt_stds = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2))
gt_pqds = np.genfromtxt(path_patch_distribution_labels,delimiter=",")[:,3:]

path = "./results/003-run/tests_%s/epoch-24_test_pred.csv" % (scenario)
pred_means = np.genfromtxt(path,delimiter=",",usecols=(1))
pred_stds = np.genfromtxt(path,delimiter=",",usecols=(2))
pred_pqds = np.genfromtxt(path,delimiter=",")[:,3:]



# Basic Plots
i = 42
gt_means[i]
sample_names[i]
plt.rcParams.update({'figure.figsize':(5,5), 'figure.dpi':100})
plt.gca().set(title="%s" % sample_names[i].split("/")[-1], ylabel='score', xlabel='partial PSNR'); plt.plot(gt_pqds[i]/gt_pqds[i].sum()); plt.plot(pred_pqds[i])

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


plt.gca().set(title="distribution of standard deviations, red=predicted PQD, blue=gt", ylabel='Frequency'); plt.hist(pred_stds, bins=100, color='red', range=(0,30), alpha = 0.5); plt.hist(gt_stds, bins=100, color='blue', range=(0,30), alpha = 0.5); plt.savefig("/Users/ralf/Desktop/stdhist.png")

plt.gca().set(title="distribution of mean, red=predicted PQD, blue=gt", ylabel='Frequency'); plt.ylim(0,25); plt.hist(pred_means, bins=100, color='red', range=(0,150), alpha = 0.5); plt.hist(gt_means, bins=100, color='blue', range=(0,150), alpha = 0.5); plt.savefig("/Users/ralf/Desktop/meanhist.png")


## MSE Scatter plots

plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.gca().set(title="MSE of ANF Model vs Ground Truth, color coded by VMAF Score", ylabel='Predicted', xlabel='Ground Truth'); plt.ylim(0,100); plt.xlim(0,100); plt.scatter(gt_means, pred_means, c=vmaf_scores); plt.colorbar(); plt.savefig("/Users/ralf/Desktop/MSE.png")



plt.gca().set(title="MSE of ANF Model vs PQD Model, color coded by VMAF Score", ylabel='PQD', xlabel='ANF'); plt.ylim(0,100); plt.xlim(0,100); plt.scatter(pred_means_anf, pred_means_pqd, c=vmaf_scores); plt.colorbar(); plt.savefig("/Users/ralf/Desktop/MSE.png")




plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':100})
plt.gca().set(title="MSE models vs Ground Truth. red=pqd, blue=anf, green=anv", ylabel='Predicted', xlabel='Ground Truth'); plt.ylim(0,100); plt.xlim(0,100); plt.scatter(gt_means, pred_means_pqd, color='red', marker='+'); plt.scatter(gt_means, pred_means_anf, color='blue', marker='+'); plt.scatter(gt_means, pred_means_anv, color='green', marker='+');

plt.savefig("/Users/ralf/Desktop/MSE.png")






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

## results as MSE of each experiment
experimentkeys = ["00", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
resultkeys = ["PQD1","PQD2","ANV1","ANV2","ANF1","ANF2"]
results = np.array([[110.5937,122.0417,121.9868,117.2077,115.4500,120.2718],
[77.5845,84.9238,85.6827,77.9098,74.3035,94.1044],
[111.5058,103.0548,109.3072,110.5940,115.5720,113.9274],
[140.6993,178.9930,170.8900,161.9356,154.1484,152.2945],
[63.1799,72.6795,83.2724,81.6832,68.9677,86.7477],
[106.9395,104.3602,110.1706,100.9691,109.4638,109.0861],
[149.2010,179.6596,164.6004,163.0855,156.3169,158.3697],
[54.0787,77.9493,75.7079,77.7112,65.1996,89.4305],
[99.6833,87.4642,99.2993,94.3899,108.3991,101.0227],
[159.5384,215.9231,203.5903,192.5412,174.0614,176.3784]])


pqd_mean = np.mean(results[:,(0,1)], axis=1)
pqd_std = np.abs(results[:,1] - results[:,0])

anv_mean = np.mean(results[:,(2,3)], axis=1)
anv_std = np.abs(results[:,2] - results[:,3])

anf_mean = np.mean(results[:,(4,5)], axis=1)
anf_std = np.abs(results[:,4] - results[:,5])

x = np.arange(len(experimentkeys))
width = 0.25

plt.rcParams.update({'figure.figsize':(18,10), 'figure.dpi':100})
plt.ylabel("MSE"); plt.xlabel("experiment"); plt.title("MSE of predictions for 2 seperately trained models on testset"); plt.xticks(x); plt.bar(x - width, pqd_mean, width, yerr=pqd_std, label="PQD"); plt.bar(x , anv_mean, width, yerr=anv_std, label="ANV"); plt.bar(x+ width, anf_mean, width, yerr=anf_std, label="ANF");  plt.legend(loc='upper left'); plt.savefig("/Users/ralf/Desktop/MSEAll.png")
