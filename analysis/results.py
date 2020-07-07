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
gt_means = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(1), skip_header=1)
gt_stds = np.genfromtxt(path_patch_distribution_labels,delimiter=",",usecols=(2), skip_header=1)
gt_pqds = np.genfromtxt(path_patch_distribution_labels,delimiter=",", skip_header=1)[:,3:]


path = "./tests_pqd/epoch-50_test_pred.csv"
pred_means = np.genfromtxt(path,delimiter=",",usecols=(1))
pred_stds = np.genfromtxt(path,delimiter=",",usecols=(2))
pred_pqds = np.genfromtxt(path,delimiter=",")[:,3:]

# plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':50})
# plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(pred_means,pred_stds)

sample_names[100]
plt.plot(gt_pqds[100])
plt.plot(pred_pqds[100])



gt_pqds[0].sum()
pred_pqds[0].sum()



pred_means = np.genfromtxt(path_patch_distribution_predictions,delimiter=",",usecols=(1))
pred_stds = np.genfromtxt(path_patch_distribution_predictions,delimiter=",",usecols=(2))
pred_pqds = np.genfromtxt(path_patch_distribution_predictions,delimiter=",")[:,3:]

pred_means2 = np.genfromtxt(path_patch_distribution_predictions2,delimiter=",",usecols=(1))
pred_stds2 = np.genfromtxt(path_patch_distribution_predictions2,delimiter=",",usecols=(2))
pred_pqds2 = np.genfromtxt(path_patch_distribution_predictions2,delimiter=",")[:,3:]

pred_means.shape
pred_stds.shape

plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':50})
plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(pred_means,pred_stds)

plt.gca().set(title="Mean vs Standard Deviation predictions"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(pred_means2,pred_stds2)

plt.gca().set(title="Mean vs Standard Deviation ground truth"); plt.xlabel("mean patch quality"); plt.ylabel("standard deviation patch quality"); plt.scatter(gt_means,gt_stds)
