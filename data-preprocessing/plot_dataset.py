import os
import random
import numpy as np
import matplotlib.pyplot as plt

# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment

path_drive_prefix = "../GTTS/"
path_pqd_labels_all = path_drive_prefix + "Labels/pqd_labels_all.csv"
path_pqd_labels_train = path_drive_prefix + "Labels/pqd_labels_train.csv"
path_pqd_labels_val = path_drive_prefix + "Labels/pqd_labels_val.csv"
path_pqd_labels_test = path_drive_prefix + "Labels/pqd_labels_test.csv"

path_anv_labels_all = path_drive_prefix + "Labels/anv_labels_all.csv"
path_anv_labels_train = path_drive_prefix + "Labels/anv_labels_train.csv"
path_anv_labels_val = path_drive_prefix + "Labels/anv_labels_val.csv"
path_anv_labels_test = path_drive_prefix + "Labels/anv_labels_test.csv"

path_anf_labels_all = path_drive_prefix + "Labels/anf_labels_all.csv"
path_anf_labels_train = path_drive_prefix + "Labels/anf_labels_train.csv"
path_anf_labels_val = path_drive_prefix + "Labels/anf_labels_val.csv"
path_anf_labels_test = path_drive_prefix + "Labels/anf_labels_test.csv"

all_samples = np.genfromtxt(path_pqd_labels_all,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()

pqd_annotations = np.genfromtxt(path_pqd_labels_all,delimiter=",")[:,1:]
anv_annotations = np.genfromtxt(path_anv_labels_all,delimiter=",")[:,1:]
anf_annotations = np.genfromtxt(path_anf_labels_all,delimiter=",")[:,1:]

pqd_annotations[0,2:].sum()
anv_annotations[0,2:].sum()
anf_annotations[:,2:].sum(axis=1)

foo = pqd_annotations[0,2:]
foo /= foo.sum()
foo.sum()







#
