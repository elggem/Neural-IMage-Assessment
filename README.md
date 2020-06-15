## NIMA: Neural IMage Assessment

This is the repository for testing NIMA on a dataset of video game transmissions. More information can be found on the Wiki.

## Implementation Details

+ This worked is based on the original implementation by [kentsyx](https://github.com/kentsyx/Neural-IMage-Assessment).

+ The code is tested with python3.6+.

## Requirements

Either ```pip install -r requirements.txt``` to install the required dependencies or use [conda](https://docs.conda.io/en/latest/) to manage your env.

## Usage

```python
python main.py --img_path /path/to/images/ --train --train_csv_file /path/to/train_labels.csv --val_csv_file /path/to/val_labels.csv --conv_base_lr 3e-4 --dense_lr 3e-3 --decay --ckpt_path /path/to/ckpts --epochs 100 --early_stoppping_patience 10
```

For inference, here the predicted score mean and std is generated. See ```predictions/``` for an example format.

```python
python test.py --model /path/to/your_model --test_csv /path/to/test_labels.csv --test_images /path/to/images --predictions /path/to/save/predictions
```
