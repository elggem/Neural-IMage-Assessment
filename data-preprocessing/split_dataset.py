# We split these labels into training and test data as follows:
# Extract all samples from N games
# Aim for 80/10/10 split

import os
import random
import numpy as np

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


all_samples = np.genfromtxt(path_pq_labels_all,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()

pqd_annotations = np.genfromtxt(path_pqd_labels_all,delimiter=",")[:,1:]
anv_annotations = np.genfromtxt(path_anv_labels_all,delimiter=",")[:,1:]
anf_annotations = np.genfromtxt(path_anf_labels_all,delimiter=",")[:,1:]

len(all_samples)

game_names = ['CSGO', 'DiabloIII', 'Dota2', 'FIFA17', 'H1Z1', 'Hearthstone', 'HeroesOfTheStorm', 'LoL', 'PU', 'ProjectCar', 'ProjectCars', 'StarCraftII', 'WoW']

test_validation_games = random.sample(game_names, 3) # LoL, CSGO, HeroesOfTheStorm

# Split all_samples
train_samples = []
test_validation_samples = []

# Extract all samples from games
for sample in all_samples:
    is_match = False
    for game in test_validation_games:
        if game in sample:
            test_validation_samples.append(sample)
            is_match = True
    if not is_match:
        train_samples.append(sample)

len(train_samples) / float(len(all_samples)) # 0.78
len(test_validation_samples) / float(len(all_samples)) # 0.21

# Split into Test / Validation Samples
random.shuffle(test_validation_samples)

test_samples = test_validation_samples[:len(test_validation_samples)/2]
validation_samples = test_validation_samples[len(test_validation_samples)/2:]

len(test_samples) / float(len(all_samples)) # 0.10
len(validation_samples) / float(len(all_samples)) # 0.10

# Write files to disk
outfile_pqd_train = open(path_pqd_labels_train, "w")
outfile_pqd_test = open(path_pqd_labels_test, "w")
outfile_pqd_val = open(path_pqd_labels_val, "w")

outfile_anv_train = open(path_anv_labels_train, "w")
outfile_anv_test = open(path_anv_labels_test, "w")
outfile_anv_val = open(path_anv_labels_val, "w")

outfile_anf_train = open(path_anf_labels_train, "w")
outfile_anf_test = open(path_anf_labels_test, "w")
outfile_anf_val = open(path_anf_labels_val, "w")

for sample in train_samples:
    dist_string_pqd = ",".join(format(x, ".8e") for x in pqd_annotations[all_samples.index(sample)])
    dist_string_anv = ",".join(format(x, ".8e") for x in anv_annotations[all_samples.index(sample)])
    dist_string_anf = ",".join(format(x, ".8e") for x in anf_annotations[all_samples.index(sample)])
    outfile_pqd_train.write("%s,%s\n" % (sample, dist_string_pqd))
    outfile_anv_train.write("%s,%s\n" % (sample, dist_string_anv))
    outfile_anf_train.write("%s,%s\n" % (sample, dist_string_anf))

for sample in test_samples:
    dist_string_pqd = ",".join(format(x, ".8e") for x in pqd_annotations[all_samples.index(sample)])
    dist_string_anv = ",".join(format(x, ".8e") for x in anv_annotations[all_samples.index(sample)])
    dist_string_anf = ",".join(format(x, ".8e") for x in anf_annotations[all_samples.index(sample)])
    outfile_pqd_test.write("%s,%s\n" % (sample, dist_string_pqd))
    outfile_anv_test.write("%s,%s\n" % (sample, dist_string_anv))
    outfile_anf_test.write("%s,%s\n" % (sample, dist_string_anf))

for sample in validation_samples:
    dist_string_pqd = ",".join(format(x, ".8e") for x in pqd_annotations[all_samples.index(sample)])
    dist_string_anv = ",".join(format(x, ".8e") for x in anv_annotations[all_samples.index(sample)])
    dist_string_anf = ",".join(format(x, ".8e") for x in anf_annotations[all_samples.index(sample)])
    outfile_pqd_val.write("%s,%s\n" % (sample, dist_string_pqd))
    outfile_anv_val.write("%s,%s\n" % (sample, dist_string_anv))
    outfile_anf_val.write("%s,%s\n" % (sample, dist_string_anf))

outfile_pqd_train.close()
outfile_anv_train.close()
outfile_anf_train.close()
outfile_pqd_test.close()
outfile_anv_test.close()
outfile_anf_test.close()
outfile_pqd_val.close()
outfile_anv_val.close()
outfile_anf_val.close()
