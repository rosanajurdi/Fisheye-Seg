'''
Created on Feb 13, 2020

@author: eljurros
'''
'''
Created on Mar 20, 2019

@author: eljurros
'''
import sys
sys.path.append('/home/eljurros/spare-workplace/WoodScape_Project/DataSet_Module')

from DataSet_Classes import WoodScapeDataSet
from torchvision import transforms
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd

import os
import shutil
import random

typ = 'ROOT'
root_path = '/media/eljurros/1 TB/Real_Data'
fold = 'FOLD_3'
if os.path.exists(os.path.join(root_path, fold)) is False:
    os.mkdir(os.path.join(root_path,  fold))
    os.mkdir(os.path.join(root_path,   fold, 'train'))
    os.mkdir(os.path.join(root_path,   fold, 'val'))
inner_arr = []
outer_arr = []
ds = WoodScapeDataSet(root_dir=root_path, typ=typ)
train_path = [os.path.join(root_path,fold,'train', 'rgb_images'), os.path.join(root_path, fold,'train', 'gtLabels')]
val_path = [os.path.join(root_path, fold,'val', 'rgb_images'), os.path.join(root_path, fold,'val', 'gtLabels')]

if os.path.exists(train_path[0]) is False:
    os.mkdir(train_path[0])
    os.mkdir(train_path[1])

nb_val = np.int(ds.__len__()*20.00/100.00)
if os.path.exists(val_path[0]) is False:
    os.mkdir(val_path[0])
    os.mkdir(val_path[1])

from tqdm import tqdm
import random
from functools import partial


val_id = random.sample(ds.filename_ids, nb_val)
val_img = [ds.filename_imgs[index] for index in val_id]
val_gt = [ds.file_name_gts[index] for index in val_id]
val_file = pd.DataFrame(data = val_img)
val_file.to_csv(os.path.join(root_path,fold,'val.txt'))
train_ids = list(set(ds.filename_ids) - set(val_id))
tr_img = [ds.filename_imgs[index] for index in train_ids]
tr_gt = [ds.file_name_gts[index] for index in train_ids]
tr_file = pd.DataFrame(data = tr_img)
tr_file.to_csv(os.path.join(root_path,fold,'train.txt'))

from multiprocessing.pool import ThreadPool
copy_to_valimgdir = partial(shutil.copy, dst = val_path[0])
copy_to_valgtdir = partial(shutil.copy, dst = val_path[1])

with ThreadPool(4) as p:
    print(p)
    p.map(copy_to_valimgdir, val_img)
with ThreadPool(4) as p:
    print(p)
    p.map(copy_to_valgtdir, val_gt)

copy_to_trimgdir = partial(shutil.copy, dst = train_path[0])
copy_to_trgtdir = partial(shutil.copy, dst = train_path[1])

with ThreadPool(5) as p:
   p.map(copy_to_trimgdir, tr_img)
   p.map(copy_to_trgtdir, tr_gt)


            
""" 
for i, patient_path in enumerate(tqdm(ds.filename_pairs)):
    patient_name = os.path.basename(patient_path[0])
    input_filename, gt_filename = patient_path[0], \
                                  patient_path[1]
    
    

    i = random.randint(1,101)
        
    if i <50 and nb_val >0:
        nb_val = nb_val - 1
        
        shutil.copy(patient_path[0], val_path[0])
        shutil.copy(patient_path[1], val_path[1])
        with open(os.path.join(os.path.join(root_path,fold,'val.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')

    else:
        shutil.copy(patient_path[0], train_path[0])
        shutil.copy(patient_path[1], train_path[1])
        with open(os.path.join(os.path.join(root_path,fold,'train.txt')), 'a') as the_file:
            the_file.write(patient_name)
            the_file.write('\n')
"""
    
        







        
