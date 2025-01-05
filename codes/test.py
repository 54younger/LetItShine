import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np
import os

import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()


def test(model, dataloader, device='cuda',fold_num = 1,mode = "BF",fusion_mode="E",model_name="resnet",run_name="default"):

    model.eval()

    labels, preds, scores, names  = [], [], [], []
    mode_name = mode
    if mode_name=="MM":
        mode_name=f"{mode_name}_{fusion_mode}_{run_name}"
    for data in tqdm(dataloader):
        X = data[mode].to(device)
        y = data['label'].float().to(device)
        f_name = data['name']

        if mode == 'MM':
            X = (X[:, :3, ...], X[:, 3:, ...]) # BF and FL
        with torch.no_grad():
            y_ = model(X).squeeze(1)
            score = torch.sigmoid(y_)

        scores.extend(score.tolist())
        labels.extend(y.tolist())
        names.extend(list(f_name))
        pred = (score > 0.5).int()
        preds.extend(pred.tolist())

    conf_matrix = metrics.confusion_matrix(labels, preds)    
    auc_score = metrics.roc_auc_score(labels, scores)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    accuracy = metrics.accuracy_score(labels, preds)
    results = {
        'Confusion Matrix': conf_matrix,
        'ROC AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    }
    # model.train()

    df = pd.DataFrame()

    df['name'] = names
    df['label'] = labels
    df['score'] = scores
    df['pred'] = preds

    df['TP'] = ((df['label'] == 1) & (df['pred'] == 1)).astype(int)
    df['TN'] = ((df['label'] == 0) & (df['pred'] == 0)).astype(int)
    df['FP'] = ((df['label'] == 0) & (df['pred'] == 1)).astype(int)
    df['FN'] = ((df['label'] == 1) & (df['pred'] == 0)).astype(int)
    print(results)
    os.makedirs(f'./excels/{model_name}_{mode_name}_{run_name}/', exist_ok = True)
    df.to_excel(f"./excels/{model_name}_{mode_name}_{run_name}/result_{fold_num}.xlsx", index=False)
    model.train()    
    return results


