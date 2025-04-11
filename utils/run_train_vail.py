import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from utils.utils import calculate_metrics
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn.functional as F


def uncertain_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum() + (prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    # Initialize metrics dictionary
    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        # Move data to device
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        mask_pred, fg_pred, bg_pred, uc_pred = model(x)

        # Calculate losses
        loss_mask = loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred, y1)
        loss_bg = loss_fn(bg_pred, y2)

        # Compute adaptive weights
        beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
        beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
        beta3 = 1 / (torch.tanh(bg_pred.sum() / (uc_pred.shape[2] * uc_pred.shape[3])) + 1e-15)
        beta1 = beta1.to(device)
        beta2 = beta2.to(device)
        beta3 = beta3.to(device)

        # Complementary loss
        preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
        probs = F.softmax(preds, dim=1)
        prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]
        loss_comp = uncertain_loss(prob_fg, prob_bg, prob_uc).to(device)
        
        # Total loss
        loss = loss_mask + beta1*loss_fg + beta2*loss_bg + beta3*loss_comp
        loss.backward()
        optimizer.step()

        # Accumulate loss
        metrics['loss'] += loss.item()

        # Calculate metrics for current batch
        batch_metrics = {
            'miou': [], 'dsc': [], 'acc': [], 'sen': [],
            'spe': [], 'pre': [], 'rec': [], 'fb': [], 'em': []
        }

        for yt, yp in zip(y1, mask_pred):
            scores = calculate_metrics(yt, yp)
            for idx, key in enumerate(batch_metrics.keys()):
                batch_metrics[key].append(scores[idx])

        # Update epoch metrics with batch averages
        for key in batch_metrics:
            metrics[key] += np.mean(batch_metrics[key])

    # Compute final epoch averages
    for key in metrics:
        metrics[key] /= len(loader)

    # Return in original format (loss, metrics_list) for compatibility
    return metrics['loss'], [
        metrics['miou'], metrics['dsc'], metrics['acc'],
        metrics['sen'], metrics['spe'], metrics['pre'],
        metrics['rec'], metrics['fb'], metrics['em']
    ]
    
def evaluate(model, loader, loss_fn, device):
    model.eval()

    metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'dsc': 0.0,
        'acc': 0.0,
        'sen': 0.0,
        'spe': 0.0,
        'pre': 0.0,
        'rec': 0.0,
        'fb': 0.0,
        'em': 0.0
    }

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            # Forward pass
            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            # Loss calculation
            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            # Beta coefficients
            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta3 = 1 / (torch.tanh(bg_pred.sum() / (uc_pred.shape[2] * uc_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)
            beta3 = beta3.to(device)

            # Uncertainty loss
            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]
            loss_comp = uncertain_loss(prob_fg, prob_bg, prob_uc).to(device)
            
            # Total loss
            loss = loss_mask + beta1*loss_fg + beta2*loss_bg + beta3*loss_comp
            metrics['loss'] += loss.item()

            # Calculate metrics for each sample
            batch_metrics = {
                'miou': [], 'dsc': [], 'acc': [], 'sen': [],
                'spe': [], 'pre': [], 'rec': [], 'fb': [], 'em': []
            }

            for yt, yp in zip(y1, mask_pred):
                scores = calculate_metrics(yt, yp)
                for idx, key in enumerate(batch_metrics.keys()):
                    batch_metrics[key].append(scores[idx])

            # Accumulate batch metrics
            for key in batch_metrics:
                metrics[key] += np.mean(batch_metrics[key])

    # Average metrics over all batches
    for key in metrics:
        metrics[key] /= len(loader)

    return metrics['loss'], [
        metrics['miou'], metrics['dsc'], metrics['acc'], 
        metrics['sen'], metrics['spe'], metrics['pre'],
        metrics['rec'], metrics['fb'], metrics['em']
    ]