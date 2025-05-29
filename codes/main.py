import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import pandas as pd 
import numpy as np
import os
import wandb
import argparse
import shutil
from data.dataloader_new import MultiModalCancerDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import train
# from utils import LayerNorm2d
from test import test
from archs import network, cafnet, embnet, fusionm4net, mmtm
from torchsampler import ImbalancedDatasetSampler
from lion_pytorch import Lion
import time
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_train_csv', default='../data')
parser.add_argument('--path_to_val_csv', default='../data')
parser.add_argument('--path_to_test_csv', default='../data')

parser.add_argument('--path_to_train_images', default='../data')
parser.add_argument('--path_to_val_images', default='../data')
parser.add_argument('--path_to_test_images', default='../data')

parser.add_argument('--checkpointBF_path', default="./runs/BF_res50_e100/checkpoint_0099.pth.tar")
parser.add_argument('--checkpointFL_path', default="./runs/FL_res50_e50/checkpoint_0074.pth.tar")
parser.add_argument('--checkpointE_path', default="./runs/E_norm/checkpoint_0034.pth.tar")
parser.add_argument('--checkpointD_path', default="./runs/imagenet_simclr/checkpoint_0023.pth.tar")

parser.add_argument('--crop_size', type=int, default=128) #change crop size to 224x224,shift
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=8e-5)
parser.add_argument('--name', default='resnet50', help='[resnet50] for single model, [cafnet, embnet, fusionm4net, mmtm] for intermediate fusion')
parser.add_argument('--mode', type=str, default='MM', help='[BF, FL, MM], MM means multi-modal inputs')
parser.add_argument('--channel', type=int, default=7, help='3 for BF, 4 for FL, 7 for MM')
parser.add_argument('--fold', type=int, default=0, help='fold number: [0, 1, 2]')
parser.add_argument('--run_name', type=str, default='early_fusion')
parser.add_argument('--fusion_mode', type=str, default='E', help='[E,L,I]')
parser.add_argument('--mixup', action='store_true', help='Whether to use the mixup augmentation')
parser.add_argument('--retrain', action='store_true', help='Whether to retrain with validation set')
parser.add_argument('--CL', action='store_true', help='Whether to use contrastive learning')
parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model')
parser.add_argument('--freeze', action='store_true', help='Whether to freeze weights')
parser.add_argument('--wd', type=float, default=0.1, help='weight decay')


def save_list_of_dicts_to_txt(file_path, dict_list):
    with open(file_path, 'w') as txtfile:
        for index, my_dict in enumerate(dict_list):
            txtfile.write(f"{index}: {my_dict}\n")

def main(args):
    start_time = time.time()

    for i in range(3):
        #i = args.fold
        mode_name = args.mode
        if mode_name =="MM":
            mode_name=f"{mode_name}_{args.fusion_mode}"
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/{args.name}_{mode_name}_{args.run_name}/fold_{i+1}'
        os.makedirs(log_path, exist_ok=True)

        path_to_train_csv = f'{args.path_to_train_csv}/train_split.csv'
        path_to_val_csv = f'{args.path_to_val_csv}/val_split.csv'
        path_to_test_csv = f'{args.path_to_test_csv}/test_split.csv'

        train_df = pd.read_csv(path_to_train_csv)
        val_df = pd.read_csv(path_to_val_csv)
        test_df = pd.read_csv(path_to_test_csv)

        if args.retrain:
            job_type = "retrain"
            train_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            job_type = "train"
      
#######################################################
        group = args.run_name
        if mode_name=='MM':
            if args.fusion_mode=='E':
                group='early'
            elif args.fusion_mode=='L':
                group="late"
            elif args.fusion_mode=='I':
                group=args.name
        else:
            group=mode_name
        if args.retrain:
            job_type = "retrain"
        else:
            job_type = "train"
        wandb.run = wandb.init(reinit=True, 
                                name=f'{args.run_name}_fold_{i+1}', 
                                project="xxx",
                                group=group,
                                job_type="job_type",
                                config={
                                "learning_rate": args.lr,
                                "architecture": args.name,
                                "epochs": args.epochs,
                                "weight_decay": args.wd,
                                "crop_size":args.crop_size,
                                "log_name":args.run_name,
                                "batch_size":args.batch_size,
                                "mixup": bool(args.mixup),
                                "retrain": bool(args.retrain),
                                "CL": bool(args.CL),
                                },)
#######################################################
        train_dataset = MultiModalCancerDataset(
            args.path_to_train_images, train_df, mode=args.mode, split='train', size=args.crop_size)
        val_dataset = MultiModalCancerDataset(
            args.path_to_val_images, val_df, mode=args.mode, split='val', size=args.crop_size)
        test_dataset = MultiModalCancerDataset(
            args.path_to_test_images, test_df, mode=args.mode, split='test', size=args.crop_size)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        
        checkpoint_BF, checkpoint_FL, checkpoint_E, checkpoint_D = None, None, None, None
        if args.CL:
            if os.path.exists(args.checkpointBF_path):
                checkpoint_BF = torch.load(args.checkpointBF_path)
            if os.path.exists(args.checkpointFL_path):
                checkpoint_FL = torch.load(args.checkpointFL_path)
            if os.path.exists(args.checkpointE_path):
                checkpoint_E = torch.load(args.checkpointE_path)
            if os.path.exists(args.checkpointD_path):
                print('Loading multi-modal SimCLR: ', args.checkpointD_path)
                checkpoint_D = torch.load(args.checkpointD_path)
            

        if args.mode == 'MM':
            if args.fusion_mode == 'I':
                if args.name == 'cafnet':
                    model = cafnet.CAFNet(channel=args.channel, pretrained=True, checkpoint_BF=checkpoint_BF, checkpoint_FL=checkpoint_FL)
                elif args.name == 'embnet':
                    model = embnet.EmbNet(channel=args.channel, pretrained=True, checkpoint_BF=checkpoint_BF, checkpoint_FL=checkpoint_FL)
                elif args.name == 'fusionm4net':
                    model = fusionm4net.FusionNet(pretrained=True, checkpoint_BF=checkpoint_BF, checkpoint_FL=checkpoint_FL)
                elif args.name == 'mmtm':
                    model = mmtm.MMTNet(pretrained=True, checkpoint_BF=checkpoint_BF, checkpoint_FL=checkpoint_FL)
            else:
               model = network.MultimodalNet(args.name, channel=args.channel, fusion_mode=args.fusion_mode, pretrained=args.pretrained, checkpoint_BF=checkpoint_BF, checkpoint_FL=checkpoint_FL, checkpoint_E=checkpoint_E) 
#checkpoint_D=checkpoint_D, freeze=args.freeze)
        else:
            checkpoint = checkpoint_BF if args.mode == 'BF' else checkpoint_FL
            model = network.SinglemodalNet(args.name, channel=args.channel, pretrained=args.pretrained, checkpoint=checkpoint) 

        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        print(f"Fold {i+1}:")
        train_log,val_log,test_log = train(model, optimizer, scheduler, train_dataloader, val_dataloader, use_mixup=args.mixup, mode=args.mode, epochs=args.epochs, device=device, name=args.name, fold_num=i+1,fusion_mode=args.fusion_mode, run_name=args.run_name, use_retrain=args.retrain,test_dataloader=test_dataloader)
#        save_list_of_dicts_to_txt(f'{log_path}/train_log.txt', train_log)
#        save_list_of_dicts_to_txt(f'{log_path}/val_log.txt', val_log)
#        save_list_of_dicts_to_txt(f'{log_path}/test_log.txt', test_log)
        #break

    wandb.finish()            
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert to hours, minutes, and seconds format
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the elapsed time as a string
    elapsed_time_str = f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
    print(f"Total training and test time: {elapsed_time_str}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

