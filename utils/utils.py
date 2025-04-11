
import random
import numpy as np
import torch
from sklearn.utils import shuffle
from utils.metrics import get_metrics
from thop import profile
import albumentations as A

""" Seeding the randomness. """
def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    miou, f1_or_dsc, accuracy,sensitivity, specificity,precision,recall,f_beta,e_measure=get_metrics(y_pred,y_true)
    return [miou, f1_or_dsc, accuracy,sensitivity, specificity,precision,recall,f_beta,e_measure]



#calculate params and flops
def calculate_params_flops(model,size=480,device=None,logger=None):
    input = torch.randn(1, 3, size, size).to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			
    print('params',params/1e6)			
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    print(f'flops={flops/1e9}, params={params/1e6}, Total params≈{total/1e6:.2f}M')


#transform
def get_transform():
    transform = A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])
    return transform
