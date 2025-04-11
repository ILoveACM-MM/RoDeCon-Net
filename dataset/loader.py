import numpy as np
from torch.utils.data import DataLoader
from utils.utils import print_and_save, shuffling,get_transform
from dataset.dataset import Kvasir_Datasets,Monu_Seg_Datasets,COVID_19_Datasets,ISIC2018_Datasets,BUSI_Datasets
from torch.utils.data import Dataset, DataLoader
import cv2

""" micro """
TEST,TRAIN = 1,2


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background
        
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        
        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = mask.copy()
            background = 255 - background

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples



def get_loader(datasets,batch_size,image_size,train_log_path):
    
    if datasets=='ISIC2018':
        train_data=ISIC2018_Datasets(mode=TRAIN)
        test_data=ISIC2018_Datasets(mode=TEST)
    elif datasets=='Kvasir':
        train_data=Kvasir_Datasets(mode=TRAIN)
        test_data=Kvasir_Datasets(mode=TEST)
    elif datasets=='BUSI':
        train_data=BUSI_Datasets(mode=TRAIN)
        test_data=BUSI_Datasets(mode=TEST)
    elif datasets=='COVID_19':
        train_data=COVID_19_Datasets(mode=TRAIN)
        test_data=COVID_19_Datasets(mode=TEST)
    elif datasets=='Monu_Seg':
        train_data=Monu_Seg_Datasets(mode=TRAIN)
        test_data=Monu_Seg_Datasets(mode=TEST)
    
    (train_x, train_y) = train_data.get_data()
    (valid_x, valid_y) = test_data.get_data()
    
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=get_transform())
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    
    return train_loader,valid_loader


