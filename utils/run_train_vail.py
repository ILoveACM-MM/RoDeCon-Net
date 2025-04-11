import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from utils.utils import print_and_save, shuffling, epoch_time, calculate_metrics
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn.functional as F


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum() + (prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_miou = 0.0
    epoch_dsc = 0.0
    epoch_acc = 0.0
    epoch_sen= 0.0
    epoch_spe = 0.0
    epoch_pre=0.0
    epoch_rec=0.0
    epoch_fb=0.0
    epoch_em=0.0



    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred, fg_pred, bg_pred, uc_pred = model(x)

        loss_mask = loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred, y1)
        loss_bg = loss_fn(bg_pred, y2)

        beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
        beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
        beta3 = 1 / (torch.tanh(bg_pred.sum() / (uc_pred.shape[2] * uc_pred.shape[3])) + 1e-15)
        beta1 = beta1.to(device)
        beta2 = beta2.to(device)
        beta3 = beta3.to(device)
        preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
        probs = F.softmax(preds, dim=1)
        prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

        loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
        loss_comp = loss_comp.to(device)
        loss = loss_mask + beta1*loss_fg + beta2 *loss_bg +  beta3*loss_comp
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        miou_=[]
        dsc_=[]
        acc_=[]
        sen_=[]
        spe_=[]
        pre_=[]
        rec_=[]
        fb_=[]
        em_=[]

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            miou_.append(score[0])
            dsc_.append(score[1])
            acc_.append(score[2])
            sen_.append(score[3])
            spe_.append(score[4])
            pre_.append(score[5])
            rec_.append(score[6])
            fb_.append(score[7])
            em_.append(score[8])
            
        epoch_miou += np.mean(miou_)
        epoch_dsc += np.mean(dsc_)
        epoch_acc += np.mean(acc_)
        epoch_sen += np.mean(sen_)
        epoch_spe += np.mean(spe_)
        epoch_pre += np.mean(pre_)
        epoch_rec += np.mean(rec_)
        epoch_fb += np.mean(fb_)
        epoch_em += np.mean(em_)

    epoch_loss = epoch_loss / len(loader)
    epoch_miou=epoch_miou/len(loader)
    epoch_dsc=epoch_dsc/len(loader)
    epoch_acc=epoch_acc/len(loader)
    epoch_sen=epoch_sen/len(loader)
    epoch_spe=epoch_spe/len(loader)
    epoch_pre=epoch_pre/len(loader)
    epoch_rec=epoch_rec/len(loader)
    epoch_fb=epoch_fb/len(loader)
    epoch_em=epoch_em/len(loader)

    return epoch_loss, [epoch_miou, epoch_dsc, epoch_acc, epoch_sen,epoch_spe,epoch_pre,epoch_rec,epoch_fb,epoch_em]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_miou = 0.0
    epoch_dsc = 0.0
    epoch_acc = 0.0
    epoch_sen= 0.0
    epoch_spe = 0.0
    epoch_pre=0.0
    epoch_rec=0.0
    epoch_fb=0.0
    epoch_em=0.0
    
    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta3 = 1 / (torch.tanh(bg_pred.sum() / (uc_pred.shape[2] * uc_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)
            beta3 = beta3.to(device)
            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)
            loss = loss_mask + beta1*loss_fg + beta2 *loss_bg +  beta3*loss_comp


            epoch_loss += loss.item()

            """ Calculate the metrics """
            miou_=[]
            dsc_=[]
            acc_=[]
            sen_=[]
            spe_=[]
            pre_=[]
            rec_=[]
            fb_=[]
            em_=[]

            
            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                miou_.append(score[0])
                dsc_.append(score[1])
                acc_.append(score[2])
                sen_.append(score[3])
                spe_.append(score[4])
                pre_.append(score[5])
                rec_.append(score[6])
                fb_.append(score[7])
                em_.append(score[8])

            epoch_miou += np.mean(miou_)
            epoch_dsc += np.mean(dsc_)
            epoch_acc += np.mean(acc_)
            epoch_sen += np.mean(sen_)
            epoch_spe += np.mean(spe_)
            epoch_pre += np.mean(pre_)
            epoch_rec += np.mean(rec_)
            epoch_fb += np.mean(fb_)
            epoch_em += np.mean(em_)
        
        
        epoch_loss = epoch_loss / len(loader)
        epoch_miou=epoch_miou/len(loader)
        epoch_dsc=epoch_dsc/len(loader)
        epoch_acc=epoch_acc/len(loader)
        epoch_sen=epoch_sen/len(loader)
        epoch_spe=epoch_spe/len(loader)
        epoch_pre=epoch_pre/len(loader)
        epoch_rec=epoch_rec/len(loader)
        epoch_fb=epoch_fb/len(loader)
        epoch_em=epoch_em/len(loader)
        

        return epoch_loss,  [epoch_miou, epoch_dsc, epoch_acc, epoch_sen,epoch_spe,epoch_pre,epoch_rec,epoch_fb,epoch_em]
