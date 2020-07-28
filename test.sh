#!/bin/bash

python test.py --test_images "../GTTS/Samples" \
--test_csv "../GTTS/Labels/pqd_labels_test.csv" \
--model ./results/002-run2/checkpoints_pqd/epoch-15.pth \
--predictions ./results/002-run2/tests_pqd/

python test.py --test_images "../GTTS/Samples" \
--test_csv "../GTTS/Labels/anv_labels_test.csv" \
--model ./results/002-run2/checkpoints_anv/epoch-28.pth \
--predictions ./results/002-run2/tests_anv/

python test.py --test_images "../GTTS/Samples" \
--test_csv "../GTTS/Labels/anf_labels_test.csv" \
--model ./results/002-run2/checkpoints_anf/epoch-24.pth \
--predictions ./results/002-run2/tests_anf/
