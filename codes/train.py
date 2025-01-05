import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
import os
from tqdm import tqdm
import pandas as pd 
import pickle
from torch.utils.tensorboard import SummaryWriter
import shutil
import wandb
import torchvision.utils as tvutils
from test import test

def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, w_a, w_b):
    return lam * criterion(pred, y_a, weight=w_a) + (1 - lam) * criterion(pred, y_b, weight=w_b)
    # return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, use_mixup=False, mode='BF', epochs=100, device='cuda', name='logs',fold_num=1,fusion_mode = 'E',run_name="default",use_retrain=False,test_dataloader=None):
    mode_name = mode
    if mode=="MM":
        mode_name = f"{mode}_{fusion_mode}"
    best_acc, best_epoch = 0, 0
    pos_rate, neg_rate = 0.78, 0.22
    his,his2,his3 = [],[],[]
    k = 0

    print(f'start training: {mode_name}, name: {name}, run_name:{run_name}')
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            image = data[mode].to(device)
            label = data['label'].to(device).float()

            if use_mixup:
                image, label_a, label_b, lam = mixup_data(image, label, alpha=0.8, device=device)
                weight_a = torch.ones_like(label_a).float()
                weight_a[label_a==0] *= neg_rate
                weight_a[label_a==1] *= pos_rate

                weight_b = torch.ones_like(label_b).float()
                weight_b[label_b==0] *= neg_rate
                weight_b[label_b==1] *= pos_rate

            if mode == 'MM':
                image = (image[:, :3, ...], image[:, 3:, ...]) # BF and FL

            pred = model(image).squeeze(1)
#            score = torch.sigmoid(pred)
            if use_mixup:
                loss = mixup_criterion(
                    F.binary_cross_entropy_with_logits, pred, label_a, label_b, lam, weight_a, weight_b)
                    # F.cross_entropy, pred, label_a, label_b, lam, weight_a, weight_b)
            else:
                weight = torch.ones_like(label).float()
                weight[label==0] *= neg_rate
                weight[label==1] *= pos_rate
                loss = F.binary_cross_entropy_with_logits(pred, label, weight=weight)
                # loss = F.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            k+=1

            if i % 100 == 0:
                #break
             #   wandb.log({"Train/Loss": loss})
                print(f'epoch: {epoch:03d}, iter: {i:04d}, loss: {loss:.6f}')
        scheduler.step()
        #######################################
        if use_retrain==False:
            print("training set......") 
            results2 = val(model, train_dataloader, mode, device)
            his2.append(results2)  
            log_data = {
                "Train/ROC AUC": results2['ROC AUC'],
                "Train/Precision": results2["Precision"],
                "Train/Recall": results2["Recall"],
                "Train/F1 Score": results2["F1 Score"],
                "Train/Accuracy": results2["Accuracy"]
            }

            wandb.log(log_data)
      
        #######################################
            print("val set......")
            results = val(model, val_dataloader, mode, device)
            print("Results:")
            print(results)
            his.append(results)
            val_acc = results["Accuracy"]
            log_data = {"Validation/Loss": results["loss"],
            "Validation/ROC AUC": results['ROC AUC'],
            "Validation/Precision": results["Precision"],
            "Validation/Recall": results["Recall"],
            "Validation/F1 Score": results["F1 Score"],
            "Validation/Accuracy": results["Accuracy"]}
            wandb.log(log_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
            print(f'epoch: {epoch}, val_auc: {val_acc:.4f}, best_auc: {best_acc:.4f}, best_epoch: {best_epoch}')
        ###########################################
        print("test set......")
        results3 = test(model, test_dataloader, device, fold_num=fold_num, mode=mode, fusion_mode=fusion_mode, model_name=name, run_name=run_name)
        his3.append(results3)
        log_data = {"Test/ROC AUC": results3['ROC AUC'],
        "Test/Precision": results3["Precision"],
        "Test/Recall": results3["Recall"],
        "Test/F1 Score": results3["F1 Score"],
        "Test/Accuracy": results3["Accuracy"]}
        wandb.log(log_data)        
###########################################      
        torch.save(model.state_dict(), f'logs/{name}_{mode_name}_{run_name}/fold_{fold_num}/model_{epoch}.pth')
        #if use_retrain==False:
        #    if early_stopper.early_stop(results["F1 Score"]):
        #        break
    print('Training is done!')

    return his2,his,his3

def val(model, dataloader, mode='BF', device='cuda'):
    model.eval()
    pos_rate, neg_rate = 0.78, 0.22

    labels, preds, scores, names  = [], [], [], []
    for data in tqdm(dataloader):
        X = data[mode].to(device)
        y = data['label'].to(device).float()
        f_name = data['name']

        if mode == 'MM':
            X = (X[:, :3, ...], X[:, 3:, ...]) # BF and FL

        with torch.no_grad():
            y_ = model(X).squeeze(1)
            score = torch.sigmoid(y_)

        scores.extend(score.tolist())
        labels.extend(y.tolist())
        names.extend(list(f_name))
        pred = (score > 0.50).int()
        preds.extend(pred.tolist())

    conf_matrix = metrics.confusion_matrix(labels,preds)    
    auc_score = metrics.roc_auc_score(labels,scores)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    accuracy = metrics.accuracy_score(labels, preds)
    weight = torch.ones_like(y).float()
    weight[y==0] *= neg_rate
    weight[y==1] *= pos_rate
    loss = F.binary_cross_entropy_with_logits(y_, y,weight=weight)
    results = {
        'Confusion Matrix': conf_matrix,
        'ROC AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'loss':loss
    }
    model.train()

    return results

