'''
Created on Nov 7, 2021

@author: eljurros
'''
import os
from torch.utils.data import Dataset

class WoodScapeDataSet(Dataset):
    '''
    Class that allows the distribution of the Woodscape dataset into folds.
    '''
    def __init__(self, root_dir, typ=None, image_dir = 'rgb_images'):
        '''
        root_dir: main directory where the un divided data is
        typ= Usually ROOT indicating the undivided data file
        image_dir: where the image folder is
        '''
        self.root_dir = root_dir
        self.filename_pairs = []
        self.filename_imgs = []
        self.file_name_gts = []
        self.filename_ids = []
        #assert typ != None   /media/eljurros/Transcend/Decathlone/melanoma/ROOT/ImagesTr
        for root_path, _, files in os.walk(os.path.join(self.root_dir, typ, image_dir), topdown=False):
            if len(files) > 1:
                for i, file in enumerate(files):
                    patient_path = os.path.join(root_path, file)
                    input_filename = self._build_train_input_filename(root_dir, patient_path, 'rgb_images')
                    gt_filename = self._build_train_input_filename(root_dir, patient_path, 'mask')
                    self.filename_pairs.append((input_filename, gt_filename))
                    self.filename_imgs.append(input_filename)
                    self.file_name_gts.append(gt_filename)
                    self.filename_ids.append(i)
                    print(input_filename, gt_filename)

    @staticmethod
    def _build_train_input_filename(root_path, patient_path, im_type='img'):
        '''
        gets the img, gt names and locations
        '''
        basename = os.path.basename(patient_path)
        base_img_path = os.path.join(root_path,patient_path.split('/')[-3],'rgb_images')
        base_gt_path = os.path.join( root_path, patient_path.split('/')[-3],'gtLabels')
        if im_type == 'rgb_images' :
            return os.path.join(base_img_path, basename)
        elif im_type == 'mask':
            return os.path.join(base_gt_path, basename)
    
    def __len__(self):
        return len(self.filename_pairs)