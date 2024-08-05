import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import wandb
import os
import numpy as np

from data import load_dataset, CustomDataset
from train import train_epoch, evaluate
from config import SEED, DEVICE, VAL_SIZE, BATCH_SIZE, NUM_EPOCHS
from model import Model


if __name__ == '__main__':
    wandb.init(project='ASMR-TDA')

    torch.manual_seed(SEED)

    df = load_dataset()
    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=42)

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(num_classes=1)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss, train_accuracy, train_f1 = train_epoch(model, criterion, optimizer, train_loader)
        end_time = timer()
        epoch_time = end_time - start_time
        val_loss, val_accuracy, val_f1 = evaluate(model, criterion, val_loader)

        print((f'Epoch: {epoch}, Epoch Time = {epoch_time:.2f}s'))
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}', f'Train F1: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}', f'Validation F1: {val_f1:.4f}')

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'train_f1': train_f1,
                   'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_f1': val_f1, 'epoch_time': epoch_time})

    wandb.finish()
