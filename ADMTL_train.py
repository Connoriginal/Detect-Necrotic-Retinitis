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
from dataset import AdversarialDataset
from model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.125, help='alpha of task 1 loss')
    parser.add_argument('--lambda_adv', type=float, default=0.05, help='lambda for adversarial loss')
    parser.add_argument('--gamma', type=float, default=0.01, help='gamma for diff loss')
    parser.add_argument('--train_type', type=str, default='smote', help='train type')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    parser.add_argument('--save_model_path', type=str, default='./model/ADMTL/', help='save model path')
    parser.add_argument('--log_dir', type=str, default='./log/ADMTL_train/', help='log dir')
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
    train_data = AdversarialDataset(mode='train', train_type=args['train_type'])
    valid_data = AdversarialDataset(mode='valid')
    print("Data Loaded")

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False)
    print("Data loader created")

    # Create model
    model = AdversarialMTL(hidden_unit=args['hidden_unit'])
    model.to(device)
    print("Model created")

    # Criterion and optimizer
    criterion = nn.BCELoss()
    adv_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    min_loss = np.inf
    epoch_flag = 0

    # Train & Validate
    for epoch in range(args['epochs']):
        # Train
        train_losses = []
        train_labels_for_ARN = np.array([])
        train_preds_for_ARN = np.array([])
        train_labels_for_CMV = np.array([])
        train_preds_for_CMV = np.array([])
        model.train()
        for i, (feature, label_for_ARN, label_for_CMV, label_for_adv) in enumerate(tqdm(train_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)
            label_for_adv = label_for_adv.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass & Prediction
            out_for_ARN, out_for_CMV, out_for_adv, (shared, private_ARN, private_CMV) = model(feature)
            out_for_ARN = out_for_ARN.squeeze()
            out_for_CMV = out_for_CMV.squeeze()
            
            ## Task Loss
            ARN_loss = criterion(out_for_ARN.to(torch.float32), label_for_ARN.to(torch.float32))
            CMV_loss = criterion(out_for_CMV.to(torch.float32), label_for_CMV.to(torch.float32))
            task_loss = args['alpha'] * ARN_loss + (1 - args['alpha']) * CMV_loss

            ## Adversarial Loss
            adv_loss = adv_criterion(out_for_adv, label_for_adv)

            ## Diff Loss (shared & private_ARN , shared & private_CMV)
            matmul_ARN = torch.matmul(shared.transpose(1, 0), private_ARN)
            matmul_CMV = torch.matmul(shared.transpose(1, 0), private_CMV)
            
            diff_ARN_loss = torch.norm(matmul_ARN, p='fro') ** 2
            diff_CMV_loss = torch.norm(matmul_CMV, p='fro') ** 2
            diff_loss = (diff_ARN_loss + diff_CMV_loss)
            
            # total loss
            loss = task_loss + args['lambda_adv'] * adv_loss + args['gamma'] * diff_loss

            pred_for_ARN = torch.where(out_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            pred_for_CMV = torch.where(out_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Save loss & append label, pred for ARN, CMV
            train_losses.append([loss.item(), task_loss.item(), ARN_loss.item(), CMV_loss.item(), adv_loss.item(), diff_loss.item(), diff_ARN_loss.item(), diff_CMV_loss.item()])
            train_labels_for_ARN = np.append(train_labels_for_ARN, label_for_ARN.cpu().numpy())
            train_preds_for_ARN = np.append(train_preds_for_ARN, pred_for_ARN.cpu().numpy())
            train_labels_for_CMV = np.append(train_labels_for_CMV, label_for_CMV.cpu().numpy())
            train_preds_for_CMV = np.append(train_preds_for_CMV, pred_for_CMV.cpu().numpy())

        # Calculate average loss & roc_auc_score
        train_loss = np.mean(train_losses, axis=0)
        train_roc_auc_score_for_ARN = roc_auc_score(train_labels_for_ARN, train_preds_for_ARN)
        train_roc_auc_score_for_CMV = roc_auc_score(train_labels_for_CMV, train_preds_for_CMV)

        # Validation
        valid_losses = []
        valid_labels_for_ARN = np.array([])
        valid_preds_for_ARN = np.array([])
        valid_labels_for_CMV = np.array([])
        valid_preds_for_CMV = np.array([])
        model.eval()

        for i, (feature, label_for_ARN, label_for_CMV, label_for_adv) in enumerate(tqdm(valid_loader)):
            feature = feature.to(device)
            label_for_ARN = label_for_ARN.to(device)
            label_for_CMV = label_for_CMV.to(device)
            label_for_adv = label_for_adv.to(device)

            # Forward pass & Prediction
            with torch.no_grad():
                out_for_ARN, out_for_CMV, out_for_adv, (shared, private_ARN, private_CMV) = model(feature)
                out_for_ARN = out_for_ARN.squeeze()
                out_for_CMV = out_for_CMV.squeeze()

                ## Task loss
                ARN_loss = criterion(out_for_ARN.to(torch.float32), label_for_ARN.to(torch.float32))
                CMV_loss = criterion(out_for_CMV.to(torch.float32), label_for_CMV.to(torch.float32))
                task_loss = (ARN_loss + CMV_loss) / 2

                ## Adversarial loss
                adv_loss = adv_criterion(out_for_adv, label_for_adv)

                ## Diff loss
                matmul_ARN = torch.matmul(shared.transpose(1, 0), private_ARN)
                matmul_CMV = torch.matmul(shared.transpose(1, 0), private_CMV)
            
                diff_ARN_loss = torch.norm(matmul_ARN, p='fro') ** 2
                diff_CMV_loss = torch.norm(matmul_CMV, p='fro') ** 2
                diff_loss = (diff_ARN_loss + diff_CMV_loss)

                # total loss
                loss = task_loss + args['lambda_adv'] * adv_loss + args['gamma'] * diff_loss

            pred_for_ARN = torch.where(out_for_ARN > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            pred_for_CMV = torch.where(out_for_CMV > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            
            # Save loss & append label, pred for ARN, CMV
            valid_losses.append([loss.item(), task_loss.item(), ARN_loss.item(), CMV_loss.item(), adv_loss.item(), diff_loss.item(), diff_ARN_loss.item(), diff_CMV_loss.item()])
            valid_labels_for_ARN = np.append(valid_labels_for_ARN, label_for_ARN.cpu().numpy())
            valid_preds_for_ARN = np.append(valid_preds_for_ARN, pred_for_ARN.cpu().numpy())
            valid_labels_for_CMV = np.append(valid_labels_for_CMV, label_for_CMV.cpu().numpy())
            valid_preds_for_CMV = np.append(valid_preds_for_CMV, pred_for_CMV.cpu().numpy())
        
        # Calculate average loss & roc_auc_score
        valid_loss = np.mean(valid_losses, axis=0)
        valid_roc_auc_score_for_ARN = roc_auc_score(valid_labels_for_ARN, valid_preds_for_ARN)
        valid_roc_auc_score_for_CMV = roc_auc_score(valid_labels_for_CMV, valid_preds_for_CMV)
        
        # Plot loss & roc_auc_score
        plotter.scalar_plot('Total loss', 'train', train_loss[0], epoch)
        plotter.scalar_plot('Task loss', 'train', train_loss[1], epoch)
        plotter.scalar_plot('ARN loss', 'train', train_loss[2], epoch)
        plotter.scalar_plot('CMV loss', 'train', train_loss[3], epoch)
        plotter.scalar_plot('Adversarial loss', 'train', train_loss[4], epoch)
        plotter.scalar_plot('Diff loss', 'train', train_loss[5], epoch)
        plotter.scalar_plot('Diff ARN loss', 'train', train_loss[6], epoch)
        plotter.scalar_plot('Diff CMV loss', 'train', train_loss[7], epoch)

        plotter.scalar_plot('Total loss', 'valid', valid_loss[0], epoch)
        plotter.scalar_plot('Task loss', 'valid', valid_loss[1], epoch)
        plotter.scalar_plot('ARN loss', 'valid', valid_loss[2], epoch)
        plotter.scalar_plot('CMV loss', 'valid', valid_loss[3], epoch)
        plotter.scalar_plot('Adversarial loss', 'valid', valid_loss[4], epoch)
        plotter.scalar_plot('Diff loss', 'valid', valid_loss[5], epoch)
        plotter.scalar_plot('Diff ARN loss', 'valid', valid_loss[6], epoch)
        plotter.scalar_plot('Diff CMV loss', 'valid', valid_loss[7], epoch)

        
        plotter.overlap_plot('Total loss',{'train': train_loss[0], 'valid': valid_loss[0]}, epoch)
        plotter.overlap_plot('Task loss',{'train': train_loss[1], 'valid': valid_loss[1]}, epoch)
        plotter.overlap_plot('ARN loss',{'train': train_loss[2], 'valid': valid_loss[2]}, epoch)
        plotter.overlap_plot('CMV loss',{'train': train_loss[3], 'valid': valid_loss[3]}, epoch)
        plotter.overlap_plot('Adversarial loss',{'train': train_loss[4], 'valid': valid_loss[4]}, epoch)
        plotter.overlap_plot('Diff loss',{'train': train_loss[5], 'valid': valid_loss[5]}, epoch)
        plotter.overlap_plot('Diff ARN loss',{'train': train_loss[6], 'valid': valid_loss[6]}, epoch)
        plotter.overlap_plot('Diff CMV loss',{'train': train_loss[7], 'valid': valid_loss[7]}, epoch)


        plotter.scalar_plot('roc_auc_score(ARN)', 'train', train_roc_auc_score_for_ARN, epoch)
        plotter.scalar_plot('roc_auc_score(ARN)', 'valid', valid_roc_auc_score_for_ARN, epoch)
        plotter.overlap_plot('roc_auc_score(ARN)',{'train': train_roc_auc_score_for_ARN, 'valid': valid_roc_auc_score_for_ARN}, epoch)
        plotter.scalar_plot('roc_auc_score(CMV)', 'train', train_roc_auc_score_for_CMV, epoch)
        plotter.scalar_plot('roc_auc_score(CMV)', 'valid', valid_roc_auc_score_for_CMV, epoch)
        plotter.overlap_plot('roc_auc_score(CMV)',{'train': train_roc_auc_score_for_CMV, 'valid': valid_roc_auc_score_for_CMV}, epoch)

        # Print loss & roc_auc_score
        print('--------------------------------------------------------------------------------')
        print('Epoch: {}/{}'.format(epoch + 1, args['epochs']))
        print('Train Total Loss: {:.4f}, Task Loss: {:.4f}, ARN Loss: {:.4f}, CMV Loss: {:.4f}, Adversarial Loss: {:.4f}, Diff Loss: {:.4f}, Diff ARN Loss: {:.4f}, Diff CMV Loss: {:.4f}'.format(train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], train_loss[6], train_loss[7])) 
        print('Train ROC-AUC-Score(ARN): {:.4f}, ROC-AUC-Score(CMV): {:.4f}'.format(train_roc_auc_score_for_ARN, train_roc_auc_score_for_CMV))
        print('\nValid Total Loss: {:.4f}, Task Loss: {:.4f}, ARN Loss: {:.4f}, CMV Loss: {:.4f}, Adversarial Loss: {:.4f}, Diff Loss: {:.4f}, Diff ARN Loss: {:.4f}, Diff CMV Loss: {:.4f}'.format(valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], valid_loss[6], valid_loss[7]))
        print('Valid ROC-AUC-Score(ARN): {:.4f}, ROC-AUC-Score(CMV): {:.4f}'.format(valid_roc_auc_score_for_ARN, valid_roc_auc_score_for_CMV))

        # Save model
        if valid_loss[0] < min_loss:
            min_loss = valid_loss[0]
            epoch_flag = epoch
            torch.save(model.state_dict(), os.path.join(args['save_model_path'], '{}.pth'.format(args['train_type'])))
            print("Model saved")
        print('--------------------------------------------------------------------------------')


    print('Model saved at epoch {}'.format(epoch_flag + 1))
        
if __name__ == '__main__':
    main()

