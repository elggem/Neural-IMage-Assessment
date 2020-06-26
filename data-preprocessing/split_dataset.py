# We then split these labels into training and test data as follows:
 # Extract all samples from the game "CSGO" and save in test set, also extract one/two/three samples from each of the other games and save in test set.
# Extract one/two/three samples from each of the other games and save in validation set.

import os
import random
import numpy as np

# For running in Editor on Desktop
# cd /Users/ralf/Documents/github/_TUBForks/Neural-IMage-Assessment

path_drive_prefix = "../GTTS/"
path_pq_labels_all = path_drive_prefix + "Labels/pq_labels_all.csv"
path_pq_labels_train = path_drive_prefix + "Labels/pq_labels_train.csv"
path_pq_labels_val = path_drive_prefix + "Labels/pq_labels_val.csv"
path_pq_labels_test = path_drive_prefix + "Labels/pq_labels_test.csv"

all_samples = np.genfromtxt(path_pq_labels_all,delimiter=",",dtype=str,encoding='utf-8',usecols=(0)).tolist()
all_annotations = np.genfromtxt(path_pq_labels_all,delimiter=",")[:,1:]

game_names = ['CSGO', 'DiabloIII', 'Dota2', 'FIFA17', 'H1Z1', 'Hearthstone', 'HeroesOfTheStorm', 'LoL', 'PU', 'ProjectCar', 'ProjectCars', 'StarCraftII', 'WoW']

# Split all_samples
train_samples = []
val_samples = []
test_samples = []

# Extract all samples from CSGO game
for sample in all_samples:
    if 'CSGO' in sample:
        test_samples.append(sample)
    else:
        train_samples.append(sample)

# For each game take N random examples for testset and M random examples for validation set, leave the rest in training set.

test_samples
train_samples
