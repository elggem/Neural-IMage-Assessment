#!/bin/bash

python test.py --test_images "../GTTS/Samples" \
--test_csv "../GTTS/Labels/$1_labels_test.csv" \
--model ./checkpoints_$1/epoch-15.pth \
--workers 40 \
--out ./tests_$1/out \
--predictions ./tests_$1/predictions
