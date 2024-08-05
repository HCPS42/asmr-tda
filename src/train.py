import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


from config import DEVICE


def train_epoch(model, criterion, optimizer, data_loader):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    epoch_size = 0
    all_labels = []
    all_preds = []

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

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    loss = running_loss / epoch_size
    accuracy = running_corrects.double() / epoch_size
    f1 = f1_score(all_labels, all_preds)

    return loss, accuracy, f1

def evaluate(model, criterion, data_loader):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    epoch_size = 0
    all_labels = []
    all_preds = []

    for inputs, labels in tqdm(data_loader, desc='Validation'):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)

        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5

        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        epoch_size += inputs.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    loss = running_loss / epoch_size
    accuracy = running_corrects.double() / epoch_size
    f1 = f1_score(all_labels, all_preds)

    return loss, accuracy, f1
