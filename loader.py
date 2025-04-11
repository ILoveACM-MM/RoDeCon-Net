import sys
import os

TEST,TRAIN = 1,2

sys.path.append(os.getcwd())

from utils.transforms import Test_Transformer, Train_Transformer
from torch.utils.data import DataLoader

from dataset.dataset import ISIC2018_Datasets,PH2_Datasets,BUSI_Datasets,Kvasir_Datasets,COVID_19_Datasets,CVC_ClinkDB_Datasets,Monu_Seg_Datasets
def get_loader(datasets,batch_size,image_size,mode):
    if mode==TRAIN:
        transformer=Train_Transformer(image_size)
    else:
        transformer=Test_Transformer(image_size)
        batch_size=1
    if datasets=='ISIC2018':
        dataset=ISIC2018_Datasets(mode=mode,transformer=transformer)
    elif datasets=='PH2':
        dataset=PH2_Datasets(mode=mode,transformer=transformer)
    elif datasets=='Kvasir':
        dataset=Kvasir_Datasets(mode=mode,transformer=transformer)
    elif datasets=='BUSI':
        dataset=BUSI_Datasets(mode=mode,transformer=transformer)
    elif datasets=='COVID_19':
        dataset=COVID_19_Datasets(mode=mode,transformer=transformer)
    elif datasets=='CVC_ClinkDB':
        dataset=CVC_ClinkDB_Datasets(mode=mode,transformer=transformer)
    elif datasets=='Monu_Seg':
        dataset=Monu_Seg_Datasets(mode=mode,transformer=transformer)

        
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=0,
                        drop_last=True)
    return loader