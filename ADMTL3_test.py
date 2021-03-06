import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from plotter import TensorboardPlotter
from dataset import Adversarial3Dataset
from model import *


import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    parser.add_argument('--load_model_path', type=str, default='./model/ADMTL/smote_34.pth', help='load model path')
    parser.add_argument('--log_dir', type=str, default='./log/ADMTL3_fs_test/', help='log dir')
    args = vars(parser.parse_args())

    # Fix random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Plotter
    plotter = TensorboardPlotter(args['log_dir'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    # Create dataset
    test_data = Adversarial3Dataset(mode='test')

    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    # Load model
    model = AdversarialMTL3(hidden_unit=args['hidden_unit'])
    model.to(device)
    model.load_state_dict(torch.load(args['load_model_path'], map_location=device))
    print("Model loaded")

    # Test
    model.eval()
    test_labels_for_0 = np.array([])
    test_pred_for_0 = np.array([])
    test_labels_for_ARN = np.array([])
    test_pred_for_ARN = np.array([])
    test_labels_for_CMV = np.array([])
    test_pred_for_CMV = np.array([])
    for i, (feature, label_for_0, label_for_ARN, label_for_CMV, label_for_adv) in enumerate(tqdm(test_loader)):
        feature = feature.to(device)
        label_for_0 = label_for_0.to(device)
        label_for_ARN = label_for_ARN.to(device)
        label_for_CMV = label_for_CMV.to(device)

        # Predict
        with torch.no_grad():
            output_for_0, output_for_ARN, output_for_CMV, _, _ = model(feature)
            output_for_0 = output_for_0.squeeze()
            output_for_ARN = output_for_ARN.squeeze()
            output_for_CMV = output_for_CMV.squeeze()
            pred_for_0 = torch.where(output_for_0 > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            pred_for_ARN = torch.where(output_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            pred_for_CMV = torch.where(output_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        
        # Save labels and predictions
        test_labels_for_0 = np.append(test_labels_for_0, label_for_0.cpu().numpy())
        test_pred_for_0 = np.append(test_pred_for_0, pred_for_0.cpu().numpy())
        test_labels_for_ARN = np.append(test_labels_for_ARN, label_for_ARN.cpu().numpy())
        test_pred_for_ARN = np.append(test_pred_for_ARN, pred_for_ARN.cpu().numpy())
        test_labels_for_CMV = np.append(test_labels_for_CMV, label_for_CMV.cpu().numpy())
        test_pred_for_CMV = np.append(test_pred_for_CMV, pred_for_CMV.cpu().numpy())
    
    # make excel file with prediction & true labels
    df = pd.DataFrame({'label0(true)' : test_labels_for_0,'label0(pred)' : test_pred_for_0, 'ARN(true)': test_labels_for_ARN, 'ARN(pred)': test_pred_for_ARN, 'CMV(true)': test_labels_for_CMV, 'CMV(pred)': test_pred_for_CMV})
    df.to_excel('./result/AMTL3/predictions.xlsx')


    # Calculate AUC & roc curve
    fpr_0, tpr_0, _ = roc_curve(test_labels_for_0, test_pred_for_0)
    auc_0 = auc(fpr_0, tpr_0)
    fpr_ARN, tpr_ARN, thresholds_ARN = roc_curve(test_labels_for_ARN, test_pred_for_ARN)
    auc_ARN = auc(fpr_ARN, tpr_ARN)
    fpr_CMV, tpr_CMV, thresholds_CMV = roc_curve(test_labels_for_CMV, test_pred_for_CMV)
    auc_CMV = auc(fpr_CMV, tpr_CMV)
    print("AUC_0 : ", auc_0)
    print("AUC_ARN : ", auc_ARN)
    print("AUC_CMV : ", auc_CMV)
    print('---------------------------------------------------------------------------------------------------------------------')

    # Calculate precision & recall score & f1 score
    precision_0 = precision_score(test_labels_for_0, test_pred_for_0)
    recall_0 = recall_score(test_labels_for_0, test_pred_for_0)
    f1_0 = f1_score(test_labels_for_0, test_pred_for_0)
    precision_ARN = precision_score(test_labels_for_ARN, test_pred_for_ARN)
    recall_ARN = recall_score(test_labels_for_ARN, test_pred_for_ARN)
    f1_ARN = f1_score(test_labels_for_ARN, test_pred_for_ARN)
    precision_CMV = precision_score(test_labels_for_CMV, test_pred_for_CMV)
    recall_CMV = recall_score(test_labels_for_CMV, test_pred_for_CMV)
    f1_CMV = f1_score(test_labels_for_CMV, test_pred_for_CMV)
    print("Precision_0 : ", precision_0)
    print("Recall_0 : ", recall_0)
    print("F1_0 : ", f1_0)
    print("Precision_ARN : ", precision_ARN)
    print("Recall_ARN : ", recall_ARN)
    print("F1_ARN : ", f1_ARN)
    print("Precision_CMV : ", precision_CMV)
    print("Recall_CMV : ", recall_CMV)
    print("F1_CMV : ", f1_CMV)

    # Plot roc curve (ARN)
    figure_ARN = plt.figure(figsize=(10, 10))
    plt.plot(fpr_ARN, tpr_ARN, label='ROC curve (area = %0.4f)' % auc_ARN)
    plt.plot([0, 1], [0, 1], 'k--')
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./result/AMTL3/AUC_ARN.png')
    plotter.figure_plot('ROC_curve(ARN)', figure_ARN, 0)
    plt.close(figure_ARN)

    # Plot roc curve (CMV)
    figure_CMV = plt.figure(figsize=(10, 10))
    plt.plot(fpr_CMV, tpr_CMV, label='ROC curve (area = %0.4f)' % auc_CMV)
    plt.plot([0, 1], [0, 1], 'k--')
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./result/AMTL3/AUC_CMV.png')
    plotter.figure_plot('ROC_curve(CMV)', figure_CMV, 0)
    plt.close(figure_CMV)

    # Plot roc curve (0)
    figure_0 = plt.figure(figsize=(10, 10))
    plt.plot(fpr_0, tpr_0, label='ROC curve (area = %0.4f)' % auc_0)
    plt.plot([0, 1], [0, 1], 'k--')
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./result/AMTL3/AUC_0.png')
    plotter.figure_plot('ROC_curve(0)', figure_0, 0)
    plt.close(figure_0)




if __name__ == '__main__':
    main()