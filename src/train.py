import torch
from tqdm import tqdm
import wandb

from config import DEVICE


def train_epoch(model, criterion, optimizer, data_loader):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    epoch_size = 0
    batch = 0

    for inputs, labels in tqdm(data_loader, desc='Training'):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        corrects = torch.sum(preds == labels.data)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += corrects
        epoch_size += inputs.size(0)

        if batch % 10 == 0:
            wandb.log({'train_loss': loss.item(), 'train_accuracy': corrects / inputs.size(0)})

        batch += 1

    loss = running_loss / epoch_size
    accuracy = running_corrects.double() / epoch_size

    return loss, accuracy

def evaluate(model, criterion, data_loader):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    epoch_size = 0

    for inputs, labels in tqdm(data_loader, desc='Validation'):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)

        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5

        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        epoch_size += inputs.size(0)

    loss = running_loss / epoch_size
    accuracy = running_corrects.double() / epoch_size

    return loss, accuracy
