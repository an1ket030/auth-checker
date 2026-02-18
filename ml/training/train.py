
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import yaml
import time
import copy
from ml.scripts.augment import get_train_transforms, get_val_test_transforms

def load_config(config_path="ml/training/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(config):
    # Load EfficientNet-B3
    weights = models.EfficientNet_B3_Weights.DEFAULT if config['model']['pretrained'] else None
    model = models.efficientnet_b3(weights=weights)
    
    # Replace Classifier Head
    # Original: (classifier): Sequential((0): Dropout(p=0.3, inplace=True), (1): Linear(in_features=1536, out_features=1000, bias=True))
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=config['model']['dropout']),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, config['model']['num_classes'])
    )
    return model

def train_model():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    data_dir = cfg['data']['root_dir']
    
    # Use transforms from augment.py
    data_transforms = {
        'train': get_train_transforms(),
        'val': get_val_test_transforms()
    }

    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    model = build_model(cfg)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (will be updated after unfreezing)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    num_epochs = cfg['training']['epochs']
    freeze_epochs = cfg['training']['freeze_backbone_epochs']

    # Phase 1: Freeze Backbone
    print("Freezing backbone layers...")
    for param in model.features.parameters():
        param.requires_grad = False
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Unfreeze after Phase 1
        if epoch == freeze_epochs:
            print("Unfreezing backbone layers...")
            for param in model.features.parameters():
                param.requires_grad = True
            
            # Differential Learning Rate
            params = [
                {'params': model.features.parameters(), 'lr': cfg['training']['learning_rate'] * 0.1},
                {'params': model.classifier.parameters(), 'lr': cfg['training']['learning_rate']}
            ]
            optimizer = optim.AdamW(params, weight_decay=cfg['training']['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    # Save checkpoint
                    os.makedirs(os.path.dirname(cfg['model']['save_path']), exist_ok=True)
                    torch.save(model.state_dict(), cfg['model']['save_path'])
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= cfg['training']['early_stopping_patience']:
            print("Early stopping triggered")
            break

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model()
