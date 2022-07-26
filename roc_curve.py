import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from plotter import TensorboardPlotter
from dataset import *
from model import *
import matplotlib.pyplot as plt


def test(model,test_loader,device,type) :
    model.eval()
    if type == 'STMLP' :
        test_pred = np.array([])
        test_true = np.array([])
        for i, (feature, label) in enumerate(tqdm(test_loader)):
            feature = feature.to(device)
            label = label.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(feature)
                output = output.squeeze()
                # pred = torch.where(output > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                pred = output
            
            # Save prediction
            test_pred = np.append(test_pred, pred.cpu().numpy())
            test_true = np.append(test_true, label.cpu().numpy())
        
        # Compute AUC & roc_curve
        fpr, tpr, thresholds = roc_curve(test_true, test_pred)
        auc_score = auc(fpr, tpr)
        return auc_score, fpr, tpr
    
    elif type == 'FSMTL' :
        test_labels_for_ARN = np.array([])
        test_pred_for_ARN = np.array([])
        test_labels_for_CMV = np.array([])
        test_pred_for_CMV = np.array([])
        for i, (feature, label_for_ARN, label_for_CMV) in enumerate(tqdm(test_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)

            # Predict
            with torch.no_grad():
                output_for_ARN, output_for_CMV = model(feature)
                output_for_ARN = output_for_ARN.squeeze()
                output_for_CMV = output_for_CMV.squeeze()
                # pred_for_ARN = torch.where(output_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                # pred_for_CMV = torch.where(output_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                pred_for_ARN = output_for_ARN
                pred_for_CMV = output_for_CMV
            
            # Save labels and predictions
            test_labels_for_ARN = np.append(test_labels_for_ARN, label_for_ARN.cpu().numpy())
            test_pred_for_ARN = np.append(test_pred_for_ARN, pred_for_ARN.cpu().numpy())
            test_labels_for_CMV = np.append(test_labels_for_CMV, label_for_CMV.cpu().numpy())
            test_pred_for_CMV = np.append(test_pred_for_CMV, pred_for_CMV.cpu().numpy())
        # Calculate AUC & roc curve
        fpr_ARN, tpr_ARN, thresholds_ARN = roc_curve(test_labels_for_ARN, test_pred_for_ARN)
        auc_ARN = auc(fpr_ARN, tpr_ARN)
        fpr_CMV, tpr_CMV, thresholds_CMV = roc_curve(test_labels_for_CMV, test_pred_for_CMV)
        auc_CMV = auc(fpr_CMV, tpr_CMV)
        return auc_ARN, auc_CMV, fpr_ARN, tpr_ARN, fpr_CMV, tpr_CMV
    
    elif type == 'ASPMTL' :
        test_labels_for_ARN = np.array([])
        test_pred_for_ARN = np.array([])
        test_labels_for_CMV = np.array([])
        test_pred_for_CMV = np.array([])
        for i, (feature, label_for_ARN, label_for_CMV, label_for_adv) in enumerate(tqdm(test_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)

            # Predict
            with torch.no_grad():
                output_for_ARN, output_for_CMV, _, _ = model(feature)
                output_for_ARN = output_for_ARN.squeeze()
                output_for_CMV = output_for_CMV.squeeze()
                # pred_for_ARN = torch.where(output_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                # pred_for_CMV = torch.where(output_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                pred_for_ARN = output_for_ARN
                pred_for_CMV = output_for_CMV
            
            # Save labels and predictions
            test_labels_for_ARN = np.append(test_labels_for_ARN, label_for_ARN.cpu().numpy())
            test_pred_for_ARN = np.append(test_pred_for_ARN, pred_for_ARN.cpu().numpy())
            test_labels_for_CMV = np.append(test_labels_for_CMV, label_for_CMV.cpu().numpy())
            test_pred_for_CMV = np.append(test_pred_for_CMV, pred_for_CMV.cpu().numpy())

        # plot roc_curve and find the best threshold
        fpr_ARN, tpr_ARN, thresholds_ARN = roc_curve(test_labels_for_ARN, test_pred_for_ARN)
        auc_ARN = auc(fpr_ARN, tpr_ARN)
        fpr_CMV, tpr_CMV, thresholds_CMV = roc_curve(test_labels_for_CMV, test_pred_for_CMV)
        auc_CMV = auc(fpr_CMV, tpr_CMV)
        return auc_ARN, auc_CMV, fpr_ARN, tpr_ARN, fpr_CMV, tpr_CMV

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    args = vars(parser.parse_args())

    # Fix random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    # Dictionary
    dic = {'task1': {'STMLP':[], 'FSMTL':[], 'ASPMTL':[]}, 'task2': {'STMLP':[], 'FSMTL':[], 'ASPMTL':[]}}

    # Load dataset model
    STMLP_class1 = SingleTaskMLP(hidden_unit=args['hidden_unit'], class_type=1)
    STMLP_class1.to(device)
    STMLP_class1.load_state_dict(torch.load('./model/singleTask/smote_1.pth', map_location=device))
    # Create dataset
    test_data = SingleTaskDataset(mode='test', class_type=1)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)
    auc, fpr, tpr = test(model=STMLP_class1, test_loader=test_loader, device=device, type='STMLP')
    dic['task1']['STMLP'].append([auc, fpr, tpr])


    STMLP_class2 = SingleTaskMLP(hidden_unit=args['hidden_unit'], class_type=2)
    STMLP_class2.to(device)
    STMLP_class2.load_state_dict(torch.load('./model/singleTask/smote_2.pth', map_location=device))
    # Create dataset
    test_data = SingleTaskDataset(mode='test', class_type=2)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)
    auc, fpr, tpr = test(model=STMLP_class2, test_loader=test_loader, device=device, type='STMLP')
    dic['task2']['STMLP'].append([auc, fpr, tpr])

    FSMTL = FullySharedMTL(hidden_unit=args['hidden_unit'])
    FSMTL.to(device)
    FSMTL.load_state_dict(torch.load('./model/multiTask/smote.pth', map_location=device))
    # Create dataset
    test_data = MultiTaskDataset(mode='test')
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)
    auc_ARN, auc_CMV, fpr_ARN, tpr_ARN, fpr_CMV, tpr_CMV = test(model=FSMTL, test_loader=test_loader, device=device, type='FSMTL')
    dic['task1']['FSMTL'].append([auc_ARN, fpr_ARN, tpr_ARN])
    dic['task2']['FSMTL'].append([auc_CMV, fpr_CMV, tpr_CMV])

    ASPMTL = AdversarialMTL(hidden_unit=args['hidden_unit'])
    ASPMTL.to(device)
    ASPMTL.load_state_dict(torch.load('./model/ADMTL/smote_alp005.pth', map_location=device))
    # Create dataset
    test_data = AdversarialDataset(mode='test')
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)
    auc_ARN, auc_CMV, fpr_ARN, tpr_ARN, fpr_CMV, tpr_CMV = test(model=ASPMTL, test_loader=test_loader, device=device, type='ASPMTL')
    dic['task1']['ASPMTL'].append([auc_ARN, fpr_ARN, tpr_ARN])
    dic['task2']['ASPMTL'].append([auc_CMV, fpr_CMV, tpr_CMV])

    # Plot roc_curves per task
    for task in dic:
        auc_STMLP, fpr_STMLP, tpr_STMLP = dic[task]['STMLP'][0]
        auc_FSMTL, fpr_FSMTL, tpr_FSMTL = dic[task]['FSMTL'][0]
        auc_ASPMTL, fpr_ASPMTL, tpr_ASPMTL = dic[task]['ASPMTL'][0]
        plt.figure(figsize=(5, 5))
        plt.plot(fpr_STMLP, tpr_STMLP, label='STMLP (AUC = %.3f)' % auc_STMLP, color='blue', linestyle='-')
        plt.plot(fpr_FSMTL, tpr_FSMTL, label='FSMTL (AUC = %.3f)' % auc_FSMTL, color='green', linestyle='-')
        plt.plot(fpr_ASPMTL, tpr_ASPMTL, label='ASPMTL (AUC = %.3f)' % auc_ASPMTL, color='red', linestyle='-')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig('./result/roc_curve/%s.png' % task)
        plt.close()


    

if __name__ == '__main__':
    main()