import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

TEST,TRAIN = 1,2
    
class ISIC2018_Datasets(Dataset):
    def __init__(self,mode):
        super().__init__()
        cwd='/home/xyq/Datasets'
        self.mode=mode
        gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Training_GroundTruth','ISIC2018_Task1_Training_GroundTruth')
        images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Training_Input','ISIC2018_Task1-2_Training_Input')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        random.shuffle(self.data)
        
        if mode==TRAIN:
            self.data=self.data[:2075]
        elif mode==TEST:
            self.data=self.data[2075:2594]
        print(len(self.data))
    
    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks



class BUSI_Datasets(Dataset):
    def __init__(self,mode):
        super().__init__()
        self.mode='/home/xyq/Datasets'
        cwd=os.getcwd()
        data_path_1=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','benign')
        data_path_2=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','malignant')
        data_path_3=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','normal')
        benign_list=sorted(os.listdir(data_path_1))
        malignant_list=sorted(os.listdir(data_path_2))
        norm_list=sorted(os.listdir(data_path_3))

        benign_image_list=[item for item in benign_list if ").png" in item]
        benign_gt_list=[item for item in benign_list if "mask.png" in item]

        malignant_image_list=[item for item in malignant_list if ").png" in item]
        malignant_gt_list=[item for item in malignant_list if "mask.png" in item]

        norm_image_list=[item for item in norm_list if ").png" in item]
        norm_gt_list=[item for item in norm_list if "mask.png" in item]
        print(len(norm_gt_list))
        
        self.data=[]
        for i in range(len(benign_image_list)):
            image_path=data_path_1+'/'+benign_image_list[i]
            mask_path=data_path_1+'/'+benign_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(malignant_image_list)):
            image_path=data_path_2+'/'+malignant_image_list[i]
            mask_path=data_path_2+'/'+malignant_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(norm_image_list)):
            image_path=data_path_3+'/'+norm_image_list[i]
            mask_path=data_path_3+'/'+norm_gt_list[i]
            self.data.append([image_path, mask_path])
        
        random.shuffle(self.data)
        print(len(self.data))
        print('打乱')
        
        if mode==TRAIN:
            self.data=self.data[:624]
        if mode==TEST:
            self.data=self.data[624:780]
        
        print(len(self.data))



    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks


class Kvasir_Datasets():
    def __init__(self,mode):
        cwd=os.getcwd()
        gts_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','masks')
        images_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','images')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "jpg" in item]
        data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            data.append([image_path, mask_path])
        random.shuffle(data)
        if mode==TRAIN:
            data=data[:880]
        elif mode==TEST:
            data=data[880:1000]
        self.data=data
    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks


class COVID_19_Datasets(Dataset):
    def __init__(self,mode):
        super().__init__()
        cwd='/home/xyq/Datasets'
        self.mode=mode
        gts_path=os.path.join(cwd,'data','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','masks')
        images_path=os.path.join(cwd,'data','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','images')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        random.shuffle(self.data)

        if mode==TRAIN:
            self.data=self.data[:716]
        elif mode==TEST:
            self.data=self.data[716:894]

        print(len(self.data))
    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks




import tifffile as tiff
class Monu_Seg_Datasets(Dataset):
    def __init__(self,mode):
        super().__init__()
        cwd='/home/xyq/Datasets'
        self.mode=mode

        gts_path=os.path.join(cwd,'data','Monu_Seg','archive','kmms_test','kmms_test','masks')
        images_path=os.path.join(cwd,'data','Monu_Seg','archive','kmms_test','kmms_test','images')
     
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        
        gts_path_=os.path.join(cwd,'data','Monu_Seg','archive','kmms_training','kmms_training','masks')
        images_path_=os.path.join(cwd,'data','Monu_Seg','archive','kmms_training','kmms_training','images')
        
        images_list_=sorted(os.listdir(images_path_))
        images_list_ = [item for item in images_list_ if "tif" in item]
        gts_list_=sorted(os.listdir(gts_path_))
        gts_list_ = [item for item in gts_list_ if "png" in item]

        for i in range(len(images_list_)):
            image_path=os.path.join(images_path_,images_list_[i])
            mask_path=os.path.join(gts_path_,gts_list_[i])
            self.data.append([image_path, mask_path])
        
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:59]
        elif mode==TEST:
            self.data=self.data[59:74]
        print(len(self.data))

    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks