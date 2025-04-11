import os
import time
import datetime
import numpy as np
import torch
from utils.utils import print_and_save, epoch_time,my_seeding
from network.model import RoDeConNet
from utils.metrics import DiceBCELoss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from utils.run_train_vail import train, evaluate
import argparse
from utils.utils import calculate_params_flops
from dataset.loader import get_loader



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="BUSI",
        help="input datasets name including ISIC2018, PH2, Kvasir, BUSI, COVID_19,CVC_ClinkDB,Monu_Seg",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default="12",
        help="input batch_size",
    )
    parser.add_argument(
        "--imagesize",
        type=int, 
        default=256,
        help="input image resolution.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="input log folder: ./log",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="the checkpoint path of last model",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu_id:",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random configure:",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="end epoch",
    )
    parser.add_argument(
        "--out_channels",
        type=list,
        default=[64, 256, 512, 1024],
        help="out_channels",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="out_channels",
    )
    parser.add_argument(
        "--esp",
        type=int,
        default=200,
        help="out_channels",
    )
    parser.add_argument(
        "--window_config",
        type=list,
        default=[3,1,1],
        help="window_config=[window size, window padding, window stride]",
    )
    parser.add_argument(
        "--align_dim",
        type=int,
        default=128,
        help="align_dim",
    )
    return parser.parse_args()



if __name__ == "__main__":
    """ micro """
    TEST,TRAIN = 1,2
    
    args = parse_args()
    # dataset
    dataset_name = args.datasets

    # hyperparameters
    seed=args.seed
    my_seeding(seed)
    image_size = args.imagesize
    size = (image_size, image_size)
    batch_size = args.batchsize
    num_epochs = args.epoch
    lr = args.lr
    lr_backbone = args.lr
    early_stopping_patience = args.esp

    pretrained_backbone = args.checkpoint
    resume_path = args.checkpoint

    # make a folder
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{dataset_name}_{current_time}"

    # Directories
    save_dir = os.path.join(args.log, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    
    """ Dataset and loader """
    train_loader, valid_loader = get_loader(dataset_name, batch_size, image_size, train_log_path)
    
    """ Model """
    device = torch.device(f'cuda:{args.gpu}')
    model = RoDeConNet(in_channels=args.out_channels,scale=[1,2,4,8],final_channel=args.align_dim,window_size=args.window_config[0], window_padding=args.window_config[1], window_stride=args.window_config[2])
    #continuing
    if pretrained_backbone:
        saved_weights = torch.load(pretrained_backbone)
        for name, param in model.named_parameters():
            if name.startswith('backbone.layer0') or name.startswith('backbone.layer1') or name.startswith('backbone.layer2') or name.startswith(
                    'backbone.layer3'):
                param.data = saved_weights[name]

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)
    #calculate params and flops
    calculate_params_flops(model, image_size, device, train_log_path)
    
    param_groups = [
        {'params': [], 'lr': lr_backbone},
        {'params': [], 'lr': lr}
    ]
  
    for name, param in model.named_parameters():
        if name.startswith('backbone.layer0') or name.startswith('backbone.layer1') or name.startswith('backbone.layer2') or name.startswith('backbone.layer3'):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)

    assert len(param_groups[0]['params']) > 0, "Layer group is empty!"
    assert len(param_groups[1]['params']) > 0, "Rest group is empty!"

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - miou:{train_metrics[0]},dsc:{train_metrics[1]},acc:{train_metrics[2]},sen:{train_metrics[3]},spe:{train_metrics[4]},pre:{train_metrics[5]},rec:{train_metrics[6]},fb:{train_metrics[7]},em:{train_metrics[8]}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - miou:{valid_metrics[0]},dsc:{valid_metrics[1]},acc:{valid_metrics[2]},sen:{valid_metrics[3]},spe:{valid_metrics[4]},pre:{valid_metrics[5]},rec:{valid_metrics[6]},fb:{valid_metrics[7]},em:{valid_metrics[8]}\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
