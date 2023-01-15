#!/usr/bin/env python3.6
from skimage.measure import label, regionprops

print('hi')

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
import csv
import sys
from scipy.ndimage import gaussian_filter
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Multi_Organ_Seg')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Common_Scripts')
from Label_Estimate_Helper_Functions import Get_contour_characteristics
sys.path.append('/home/eljurros/spare-workplace/surface-loss-master')
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, probs2one_hot, class2one_hot, numpy_haussdorf
#root = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/npy/val'
#net_path = '/home/2017011/reljur01/surface-loss-master/data/Prostate/FOLD_2/results/clDice_with_coord/best2.pkl'
root='/media/eljurros/Transcend/Decathlone/benchmark/Spleen/FOLD_2/npy/val'
for loss in ['gmd', 'clDice',  'contour_alone' , 'contour_mse',  'gmd',  'HDDT' ,  'size' , 'surface']:
    net_path = '/media/eljurros/Transcend/Decathlone/benchmark/Spleen/FOLD_2/results/{}/best2.pkl'.format(loss)
    
    net = torch.load(net_path, map_location=torch.device('cpu'))
    #print(net)
    fieldnames = ['SLICE_ID', 'dice','haus',  'c_error']
    n_classes = 2
    n = 1
    #assert os.path.exists(os.path.join(net_path.split(os.path.basename(net_path))[0], 'predictions'))== False
    print('started with', net_path)
    exp_path = net_path.split('/best2.pkl')[0]
    name =os.path.basename(exp_path)
    folder_path = Path(exp_path, 'CSV_RESULTS')
    
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(exp_path, name)
    fold_clean_H1 = open(os.path.join(folder_path, '{}_{}_clean_NEW.csv'.format(name,n)), "w")
    fold_all_H1 = open(os.path.join(folder_path, '{}_{}_all_NEW.csv'.format(name,n)), "w")
     
    
    fold_all_H1.write(f"file, dice, haussdorf,connecterror, border-index \n")
    fold_clean_H1.write(f"file, dice, haussdorf,connecterror \n")
    
     
    path=os.path.join(net_path.split(os.path.basename(net_path))[0])
    
    pred_path = Path(path,'predictions')
    gt_path = Path(path,'gt')
    
    pred_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)
    
    def border_irregularity_index(gt):
        simple_image_1 = gt
        for sig in range(1,20):
            simple_image_1 = simple_image_1 + gaussian_filter(simple_image_1, sigma = sig)
            simple_image_1 = (simple_image_1>0.43)*1.00
        bi1_gt =1 - (simple_image_1*gt).sum()/(simple_image_1.sum()+gt.sum()- (simple_image_1*gt).sum() )
        return bi1_gt
                
    for _,_,files in os.walk(os.path.join(root, 'in_npy')): 
    
        print('walking into', os.path.join(root, 'in_npy'))
        for file in files: 
            print(file)
            image = np.load(os.path.join(root,'in_npy', file))
            gt = np.load(os.path.join(root,'gt_npy', file))        
            if len(np.unique(gt)) == 2:
                #print('infering {} of shape {} and classes {}, max {} and min {} '.format( file, image.shape, np.unique(gt), image.max(), image.min()))
                image = image.reshape(-1, 1, 256, 256)
                image = torch.tensor(image, dtype=torch.float)
                image = Variable(image, requires_grad=True)
                pred = net(image)
                pred = F.softmax(pred, dim=1).to('cpu')
                predicted_output = probs2one_hot(pred.detach())
                #print(predicted_output.to('cpu')[:,:2:].shape,class2one_hot(torch.tensor(gt).to('cpu'), n_classes).shape )
                #np.save(os.path.join(path, 'predictions', '{}'.format(file)), pred.to('cpu').detach().numpy())
                #dice = dice_coef(predicted_output.to('cpu'), class2one_hot(torch.tensor(gt).to('cpu'), n_classes))[:,n,]
                dice = dice_coef(predicted_output.to('cpu'), class2one_hot(torch.tensor(gt).to('cpu'), n_classes))[:,n,]
    
                hauss = haussdorf(predicted_output, class2one_hot(torch.tensor(gt), n_classes))[:,n,]
                
                BI_index = (np.abs(border_irregularity_index(gt)- 
                                  border_irregularity_index(np.argmax(pred.squeeze().detach().numpy(),axis=0)))/border_irregularity_index(gt))*100
                print(BI_index)
                '''
                fig, ax = plt.subplots()
                ax.imshow(np.argmax(predicted_output.detach().numpy(), axis=1)[0], cmap=plt.cm.gray)
                #r, contours= Get_contour_characteristics(np.argmax(predicted_output.detach().numpy(), axis=1)[0])
                g, contours = Get_contour_characteristics(np.array(gt).round())
                total_summ = 0
                
                for n, contour in enumerate(contours):
                    ax.plot(contour[:, 1].astype(int), contour[:, 0].astype(int),color='red', linewidth=-1)
    
                    ax.axis('image')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    #plt.show()
                
                plt.savefig(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])))
                #plt.imsave(os.path.join(path, 'predictions', '{}.png'.format(file.split('.npy')[0])), np.argmax(predicted_output,1)[0]) 
                #plt.imsave(os.path.join(path, 'gt', '{}.png'.format(file.split('.npy')[0])), gt)            
                '''
                gt_label = len(np.unique(label((gt==n)*1.00)))
                pred_label = len(np.unique(label(predicted_output[:,n:][0][0])))
                gt_label = len(np.unique(label(class2one_hot(torch.tensor(gt), n_classes)[0][n])))
                error = np.abs(pred_label - gt_label)
    
                '''
                print(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
    
                fold_all_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
    
                if len(np.unique(gt)) == 2:
                    fold_clean_H1.write(f"{file}, {np.float(dice[0][1])}, {np.float(hauss[0][1])},{np.float(error)} \n")
                    fold_clean_H1.flush()
                #folders.write("hi")
                fold_all_H1.flush()
                '''
                print(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)}, {np.float(BI_index)} \n")
    
                fold_all_H1.write(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)}, {np.float(BI_index)} \n")
    
                if len(np.unique(gt)) != 1:
                    fold_clean_H1.write(f"{file}, {np.float(dice[0])}, {np.float(hauss[0])},{np.float(error)} \n")
                    fold_clean_H1.flush()
                #folders.write("hi")
                fold_all_H1.flush()
            
  
        

        
        
        
        


