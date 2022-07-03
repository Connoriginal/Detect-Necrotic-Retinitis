import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from plotter import TensorboardPlotter
from dataset import MultiTaskDataset
from model import FullySharedMTL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_type', type=str, default='smote', help='train type')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    parser.add_argument('--save_model_path', type=str, default='./model/multiTask', help='save model path')
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    args = vars(parser.parse_args())

    # Fix random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Plotter
    plotter = TensorboardPlotter(args['log_dir'])

    # Plot hyper parameters
    plotter.hparams_plot(args, {})

    # Device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)


    # Create dataset
    train_data = MultiTaskDataset(mode='train', train_type=args['train_type'])
    valid_data = MultiTaskDataset(mode='valid')
    print("Data Loaded")

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False)
    print("Data loader created")

    # Create model
    model = FullySharedMTL(hidden_unit=args['hidden_unit'])
    model.to(device)
    print("Model created")

    # Criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    min_loss = np.inf

    # Train & Validate
    for epoch in range(args['epochs']):
        # Train
        train_losses = []
        train_labels_for_ARN = np.array([])
        train_preds_for_ARN = np.array([])
        train_labels_for_CMV = np.array([])
        train_preds_for_CMV = np.array([])
        model.train()
        for i, (feature, label_for_ARN, label_for_CMV) in enumerate(tqdm(train_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass & Prediction
            out_for_ARN, out_for_CMV = model(feature)
            out_for_ARN = out_for_ARN.squeeze()
            out_for_CMV = out_for_CMV.squeeze()
            ARN_loss = criterion(out_for_ARN.to(torch.float32), label_for_ARN.to(torch.float32))
            CMV_loss = criterion(out_for_CMV.to(torch.float32), label_for_CMV.to(torch.float32))
            loss = ARN_loss + CMV_loss
            pred_for_ARN = torch.where(out_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            pred_for_CMV = torch.where(out_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Save loss & append label, pred for ARN, CMV
            train_losses.append(loss.item())
            train_labels_for_ARN = np.append(train_labels_for_ARN, label_for_ARN.cpu().numpy())
            train_preds_for_ARN = np.append(train_preds_for_ARN, pred_for_ARN.cpu().numpy())
            train_labels_for_CMV = np.append(train_labels_for_CMV, label_for_CMV.cpu().numpy())
            train_preds_for_CMV = np.append(train_preds_for_CMV, pred_for_CMV.cpu().numpy())

        # Calculate average loss & roc_auc_score
        train_loss = np.mean(train_losses)
        train_roc_auc_score_for_ARN = roc_auc_score(train_labels_for_ARN, train_preds_for_ARN)
        train_roc_auc_score_for_CMV = roc_auc_score(train_labels_for_CMV, train_preds_for_CMV)

        # Validation
        valid_losses = []
        valid_labels_for_ARN = np.array([])
        valid_preds_for_ARN = np.array([])
        valid_labels_for_CMV = np.array([])
        valid_preds_for_CMV = np.array([])
        model.eval()

        for i, (feature, label_for_ARN, label_for_CMV) in enumerate(tqdm(valid_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)

            # Forward pass & Prediction
            with torch.no_grad():
                out_for_ARN, out_for_CMV = model(feature)
                out_for_ARN = out_for_ARN.squeeze()
                out_for_CMV = out_for_CMV.squeeze()
                ARN_loss = criterion(out_for_ARN.to(torch.float32), label_for_ARN.to(torch.float32))
                CMV_loss = criterion(out_for_CMV.to(torch.float32), label_for_CMV.to(torch.float32))
                loss = ARN_loss + CMV_loss
                pred_for_ARN = torch.where(out_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                pred_for_CMV = torch.where(out_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            
            # Save loss & append label, pred for ARN, CMV
            valid_losses.append(loss.item())
            valid_labels_for_ARN = np.append(valid_labels_for_ARN, label_for_ARN.cpu().numpy())
            valid_preds_for_ARN = np.append(valid_preds_for_ARN, pred_for_ARN.cpu().numpy())
            valid_labels_for_CMV = np.append(valid_labels_for_CMV, label_for_CMV.cpu().numpy())
            valid_preds_for_CMV = np.append(valid_preds_for_CMV, pred_for_CMV.cpu().numpy())
        
        # Calculate average loss & roc_auc_score
        valid_loss = np.mean(valid_losses)
        valid_roc_auc_score_for_ARN = roc_auc_score(valid_labels_for_ARN, valid_preds_for_ARN)
        valid_roc_auc_score_for_CMV = roc_auc_score(valid_labels_for_CMV, valid_preds_for_CMV)
        
        # Plot loss & roc_auc_score
        plotter.scalar_plot('loss', 'train', train_loss, epoch)
        plotter.scalar_plot('loss', 'valid', valid_loss, epoch)
        plotter.overlap_plot('loss',{'train': train_loss, 'valid': valid_loss}, epoch)
        plotter.scalar_plot('roc_auc_score(ARN)', 'train', train_roc_auc_score_for_ARN, epoch)
        plotter.scalar_plot('roc_auc_score(ARN)', 'valid', valid_roc_auc_score_for_ARN, epoch)
        plotter.overlap_plot('roc_auc_score(ARN)',{'train': train_roc_auc_score_for_ARN, 'valid': valid_roc_auc_score_for_ARN}, epoch)
        plotter.scalar_plot('roc_auc_score(CMV)', 'train', train_roc_auc_score_for_CMV, epoch)
        plotter.scalar_plot('roc_auc_score(CMV)', 'valid', valid_roc_auc_score_for_CMV, epoch)
        plotter.overlap_plot('roc_auc_score(CMV)',{'train': train_roc_auc_score_for_CMV, 'valid': valid_roc_auc_score_for_CMV}, epoch)

        # Print loss & roc_auc_score
        print('--------------------------------------------------------------------------------')
        print('Epoch: {}/{}'.format(epoch + 1, args['epochs']))
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Train ROC-AUC-Score(ARN): {:.4f}'.format(train_roc_auc_score_for_ARN))
        print('Train ROC-AUC-Score(CMV): {:.4f}'.format(train_roc_auc_score_for_CMV))
        print('\nValid Loss: {:.4f}'.format(valid_loss))
        print('Valid ROC-AUC-Score(ARN): {:.4f}'.format(valid_roc_auc_score_for_ARN))
        print('Valid ROC-AUC-Score(CMV): {:.4f}'.format(valid_roc_auc_score_for_CMV))

        # Save model
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args['save_model_path'], '{}_{}.pth'.format(args['train_type'],epoch)))
            print("Model saved")
        print('--------------------------------------------------------------------------------')



        
if __name__ == '__main__':
    main()

