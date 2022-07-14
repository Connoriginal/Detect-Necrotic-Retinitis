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
from dataset import SingleTaskDataset
from model import SingleTaskMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_type', type=str, default='smote', help='train type')
    parser.add_argument('--class_type', type=int, default=1, help='class_type')
    parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
    parser.add_argument('--save_model_path', type=str, default='./model/singleTask', help='save model path')
    parser.add_argument('--log_dir', type=str, default='./log/singleTask', help='log dir')
    args = vars(parser.parse_args())

    # Fix random seed
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Plotter
    plotter = TensorboardPlotter(args['log_dir'])

    # Plot hyper parameters
    plotter.hparams_plot(args, {})

    # Device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)


    # Create dataset
    train_data = SingleTaskDataset(mode='train', train_type=args['train_type'], class_type=args['class_type'])
    valid_data = SingleTaskDataset(mode='valid', class_type=args['class_type'])
    print("Data Loaded")

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False)
    print("Data loader created")

    # Create model
    model = SingleTaskMLP(hidden_unit=args['hidden_unit'], class_type=args['class_type'])
    model.to(device)
    print("Model created")

    # Criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    min_loss = np.inf
    epoch_flag = 0

    # Train & Validate
    for epoch in range(args['epochs']):
        # Train
        train_losses = []
        train_labels = np.array([])
        train_preds = np.array([])
        model.train()
        for i, (feature, label) in enumerate(tqdm(train_loader)):
            feature = feature.to(device)
            label = label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward & Predict(binary classification)
            output = model(feature)
            output = output.squeeze()
            loss = criterion(output.to(torch.float32), label.to(torch.float32))
            pred = torch.where(output > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            

            # Backward
            loss.backward()
            optimizer.step()

            # Save loss & append label & pred
            train_losses.append(loss.item())
            train_labels = np.append(train_labels, label.cpu().numpy())
            train_preds = np.append(train_preds, pred.cpu().numpy())
            
        # Calculate average loss & roc_auc_score
        train_loss = np.mean(train_losses)
        train_roc_auc_score = roc_auc_score(train_labels, train_preds)
        
        # Validation
        valid_losses = []
        valid_labels = np.array([])
        valid_preds = np.array([])
        model.eval()

        for i, (feature, label) in enumerate(tqdm(valid_loader)):
            feature = feature.to(device)
            label = label.to(device)
            # Forward
            with torch.no_grad():
                output = model(feature)
                output = output.squeeze()
                loss = criterion(output.to(torch.float32), label.to(torch.float32))
                pred = torch.where(output > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))

            # Save loss & append label & pred
            valid_losses.append(loss.item())
            valid_labels = np.append(valid_labels, label.cpu().detach().numpy())
            valid_preds = np.append(valid_preds, pred.cpu().detach().numpy())
        
        # Calculate loss & roc_auc_score
        valid_loss = np.mean(valid_losses)
        valid_roc_auc_score = roc_auc_score(valid_labels, valid_preds)
        
        # Plot loss & roc_auc_score
        plotter.scalar_plot('loss', 'train', train_loss, epoch)
        plotter.scalar_plot('loss', 'valid', valid_loss, epoch)
        plotter.overlap_plot('loss',{'train': train_loss, 'valid': valid_loss}, epoch)
        plotter.scalar_plot('roc_auc_score', 'train', train_roc_auc_score, epoch)
        plotter.scalar_plot('roc_auc_score', 'valid', valid_roc_auc_score, epoch)
        plotter.overlap_plot('roc_auc_score',{'train': train_roc_auc_score, 'valid': valid_roc_auc_score}, epoch)

        # Print loss & roc_auc_score
        print('--------------------------------------------------------------------------------')
        print('Epoch: {}/{}'.format(epoch + 1, args['epochs']))
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Valid Loss: {:.4f}'.format(valid_loss))
        print('Train ROC AUC Score: {:.4f}'.format(train_roc_auc_score))
        print('Valid ROC AUC Score: {:.4f}'.format(valid_roc_auc_score))

        # Save model
        if valid_loss < min_loss:
            min_loss = valid_loss
            epoch_flag = epoch
            torch.save(model.state_dict(), os.path.join(args['save_model_path'], '{}_{}.pth'.format(args['train_type'], args['class_type'])))
            print("Model saved")
        print('--------------------------------------------------------------------------------')

    print("Best model is {} epoch".format(epoch_flag + 1))

        
if __name__ == '__main__':
    main()

