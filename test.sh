#!/bin/bash

python test.py --test_images "../GTTS/Samples" \
--test_csv "../GTTS/Labels/$1_labels_test.csv" \
--model ./checkpoints_$1/epoch-37.pth \
--predictions ./tests_$1/
