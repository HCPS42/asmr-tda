import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import wandb
import os

from data import load_diagram_images, CustomDataset
from train import train_epoch, evaluate
from config import SEED, CHECKPOINTS_PATH, DEVICE, VAL_SIZE, BATCH_SIZE, NUM_EPOCHS


if __name__ == '__main__':
    wandb.init(project='ASMR-TDA')
    checkpoints_path = f'{CHECKPOINTS_PATH}/{wandb.run.name}'
    os.makedirs(checkpoints_path, exist_ok=True)

    torch.manual_seed(SEED)

    df = load_diagram_images()
    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=42)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(val_df, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader)
        end_time = timer()
        epoch_time = end_time - start_time
        val_loss, val_accuracy = evaluate(model, criterion, val_loader)

        print((f'Epoch: {epoch}, Epoch Time = {epoch_time:.2f}s'))
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'epoch_time': epoch_time})

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{checkpoints_path}/model_and_optimizer_epoch_{epoch}.pth')

    wandb.finish()
