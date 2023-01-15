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
from utils import IoU_coeff as IoU_coef
from pthflops import count_ops

sys.path.append('/home/eljurros/spare-workplace/WoodScape_Segmentation_Project')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
import metrics
root = '/home/2017011/reljur01/Synthetic_data/FOLD_M3/train'

net_path = '/home/2017011/reljur01/Synthetic_data/FOLD_M3/results/ResidualUNet/best.pkl'
print(net_path)
net = torch.load(net_path, map_location=torch.device('cpu')).to('cpu').module
#print(net)
fieldnames = ['SLICE_ID', 'IoU']
n_classes = 23
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

metric = metrics.IoU(24, 'woodscape_syn', ignore_index=[0,15,17,23])


k = 20
for _,_,files in os.walk(os.path.join(root, 'rgb_images')):
    for file in tqdm(files):
        #k = k - 1
        image = np.array(Image.open(os.path.join(root,'rgb_images', file)).resize((512,512)))/255.00
        image = np.transpose(image,(-1,0,1))
        #b = '/media/eljurros/Crucial X6/Synthetic_Data/Multi_view_synthetic_Woodscape_dataset/ROOT/seg'
        gt = np.array(Image.open(os.path.join(root,'gtLabels', file)).resize((512,512))) 
        #t = np.array(Image.open(os.path.join('/media/eljurros/Crucial X6/Synthetic_Data/FOLD_1/Results/DRU_Dec/best_epoch/val', file)).resize((512,512))) 
        
        if len(np.unique(gt)) >0:      
            image = image.reshape(-1, 3, 512, 512)
            image = torch.tensor(image, dtype=torch.float)
            image = Variable(image, requires_grad=True)
            
            # calculating flops 
            pred = net(image)
            c = count_ops(net, image)
            print("FLops:", c)
            pred = F.softmax(pred, dim=1)
            predicted_output = probs2one_hot(pred.detach())
            #gt = gt[gt>10] = 0
            #assert len(np.unique(gt))>=len(np.unique(predicted_output.argmax(dim=1)[0]))
            #a = (gt==10)*1.00
            #print(a.sum())
            #gt[gt >= 10] = 0
            #ground_truth = class2one_hot(torch.tensor(gt), 10)
            #np.save(os.path.join(path, 'predictions', '{}'.format(file)), pred.to('cpu').detach().numpy())
            #print(np.unique(np.argmax(predicted_output.detach().numpy(), axis=1)[0]))
            #predicted_output = predicted_output[:,:10]
            metric.add(torch.tensor(gt), predicted_output.argmax(dim=1)[0])
            
            #metric.add(predicted_output, class2one_hot(torch.tensor(gt), 10))
            #class_iou, mean_iou = metric.value()
            #print(class_iou)
            #if  k ==0:
            #    break
            break
class_iou, mean_iou = metric.value()
with open(Path(exp_path, "{}_Flops.txt".format(os.path.basename(exp_path))), 'w') as f:
                f.write('class iou: {}\n'.format(class_iou))
                f.write(str(mean_iou))
print(class_iou)
print(mean_iou)
  
        

        
        
        
        

