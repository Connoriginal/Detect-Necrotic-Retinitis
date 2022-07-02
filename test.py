import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader
from plotter import TensorboardPlotter
from dataset import SingleTaskDataset
from model import SingleTaskMLP
import matplotlib.pyplot as plt

import shap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--class_type', type=int, default=1, help='class_type')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    parser.add_argument('--load_model_path', type=str, default='./model/smote_1.pth', help='load model path')
    parser.add_argument('--log_dir', type=str, default='./log/test/', help='log dir')
    args = vars(parser.parse_args())

    # Fix random seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Plotter
    plotter = TensorboardPlotter(args['log_dir'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    # Create dataset
    test_data = SingleTaskDataset(mode='test', class_type=args['class_type'])

    # Create dataloader
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    # Load model
    model = SingleTaskMLP(hidden_unit=args['hidden_unit'], class_type=args['class_type'])
    model.to(device)
    model.load_state_dict(torch.load(args['load_model_path'], map_location=device))
    print("Model loaded")

    # Test
    model.eval()
    test_pred = np.array([])
    test_true = np.array([])
    for i, (feature, label) in enumerate(tqdm(test_loader)):
        feature = feature.to(device)
        label = label.to(device)
        
        # Predict
        with torch.no_grad():
            output = model(feature)
            output = output.squeeze()
            pred = torch.where(output > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        
        # Save prediction
        test_pred = np.append(test_pred, pred.cpu().numpy())
        test_true = np.append(test_true, label.cpu().numpy())
    

    # make excel file with prediction & true label
    df = pd.DataFrame({'pred': test_pred, 'true': test_true})
    df.to_excel('./result/pred_true_' + str(args['class_type']) + '.xlsx')


    # Compute AUC & roc_curve
    fpr, tpr, thresholds = roc_curve(test_true, test_pred)
    auc_score = auc(fpr, tpr)
    print("AUC score : ", auc_score)


    # Plot roc_curve
    figure = plt.figure()
    ## Visualize ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    ## 대각선 추가
    plt.plot([0, 1], [0, 1], 'k--')

    ## FPR X 축의 scla 0.1 단위 지정
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Plot figure & show
    plotter.figure_plot('roc_curve', figure, 0)
    plt.show()

    # SHAP
    features = test_data.features
    ## convert into Tensor
    features = torch.from_numpy(features).float().to(device)
    explainer = shap.DeepExplainer(model, features)
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, features, plot_type="bar")
    plotter.figure_plot('shap_values', shap.summary_plot(shap_values, features, plot_type="bar"), 0)


if __name__ == '__main__':
    main()