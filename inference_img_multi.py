#!/usr/bin/env python3.6
from skimage.measure import label, regionprops

print('hi')

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv
from tqdm import tqdm
from functools import partial


sys.path.append('/home/eljurros/spare-workplace/WoodScape_Segmentation_Project')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
import metrics
root = '/home/2017011/reljur01/OMNIDATA/FOLD_2/val'
net_path = '/home/2017011/reljur01/OMNIDATA/FOLD_2/results_2/DRU_EncALL/best.pkl'
print(net_path)
net = torch.load(net_path, map_location=torch.device('cpu')).to('cpu').module
#print(net)
fieldnames = ['SLICE_ID', 'IoU']
n_classes = 10
#assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False
print(net_path)
print(root)
exp_path = net_path.split('/best.pkl')[0]
name =os.path.basename(exp_path)
folder_path = Path(exp_path, 'CSV_RESULTS')

folder_path.mkdir(parents=True, exist_ok=True)
file_path = os.path.join(exp_path, name)
fold_clean_H1 = open(os.path.join(folder_path, '{}_clean.csv'.format(name)), "w")
fold_all_H1 = open(os.path.join(folder_path, '{}_all.csv'.format(name)), "w")
 

fold_all_H1.write(f"file, dice, haussdorf,connecterror \n")
fold_clean_H1.write(f"file, dice, haussdorf,connecterror \n")

 
path=os.path.join(net_path.split(os.path.basename(net_path))[0])

pred_path = Path(path,'predictions')
gt_path = Path(path,'gt')

pred_path.mkdir(parents=True, exist_ok=True)
gt_path.mkdir(parents=True, exist_ok=True)

metric = metrics.IoU(10, 'woodscape_raw', ignore_index=0)

from multiprocessing import Pool

for _,_,files in os.walk(os.path.join(root, 'rgb_images')):
    for file in tqdm(files):
        print(file)
        
        image = np.array(Image.open(os.path.join(root,'rgb_images', file)).resize((512,512)))/255.00
        image = np.transpose(image,(-1,0,1))
        gt = np.array(Image.open(os.path.join(root,'gtLabels', file)).resize((512,512))) 
        print(len(np.unique(gt)))
        if len(np.unique(gt)) >0:      
            image = image.reshape(-1, 3, 512, 512)
            image = torch.tensor(image, dtype=torch.float)
            image = Variable(image, requires_grad=True)
            pred = net(image)
            pred = F.softmax(pred, dim=1)
            predicted_output = probs2one_hot(pred.detach())
           
            metric.add(torch.tensor(gt), predicted_output.argmax(dim=1)[0])


class_iou, mean_iou = metric.value()
with open(Path(exp_path, "{}_MiOu.txt".format(os.path.basename(exp_path))), 'w') as f:
                f.write('class iou: {}\n'.format(class_iou))
                f.write(str(mean_iou))
print(class_iou)
print(mean_iou)

  
        

        
        
        
        

