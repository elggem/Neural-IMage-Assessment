import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--test_csv', type=str, help='test csv file')
parser.add_argument('--test_images', type=str, help='path to folder containing images')
parser.add_argument('--predictions', type=str, help='output file to store predictions')
args = parser.parse_args()

# Running in Hydrogen
# class Dummy(object):
#     pass
# args = Dummy()
# args.test_csv = "../GTTS/Labels/pqd_labels_test.csv"
# args.test_images = "../GTTS/Samples"
# args.model = "./checkpoints_pqd/epoch-15.pth"


base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

try:
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.model))
    print('successfully loaded model')
except:
    raise

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()

test_transform = transforms.Compose([
    transforms.Resize(256, 0),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])

test_df = pd.read_csv(args.test_csv, header=None)
test_imgs = test_df[0]
pbar = tqdm(total=len(test_imgs))

i = 0
img = test_imgs[0]

mean, std = 0.0, 0.0
for i, img in enumerate(test_imgs):
    im = Image.open(os.path.join(args.test_images, str(img)))
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.numpy().reshape(100)

    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5

    # gt = test_df[test_df[0] == img].to_numpy()[:, 3:].reshape(100, 1)
    # gt_mean = 0.0
    # for l, e in enumerate(gt, 1):
    #     gt_mean += l * e
    # print(str(img) + ' mean: %.3f | std: %.3f | GT: %.3f' % (mean, std, gt_mean))

    if not os.path.exists(args.predictions):
        os.makedirs(args.predictions)
    with open(os.path.join(args.predictions, args.model.split("/")[-1].split(".")[0]+'_test_pred.csv'), 'a') as f:
        # f.write(str(img) + ' mean: %.3f | std: %.3f | GT: %.3f\n' % (mean, std, gt_mean))
        f.write("%s,%.8e,%.8e,%s\n" % (str(img), mean, std, ",".join(format(x, ".8e") for _,x in enumerate(out, 1))))

    mean, std = 0.0, 0.0
    pbar.update()
