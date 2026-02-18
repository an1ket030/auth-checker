
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
from ml.training.train import build_model
from ml.scripts.augment import get_val_test_transforms

def load_config(config_path="ml/training/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = build_model(cfg)
    model_path = cfg['model']['save_path']
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Data Loader
    data_dir = cfg['data']['root_dir']
    test_transforms = get_val_test_transforms()
    
    try:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
        class_names = test_dataset.classes
    except Exception as e:
        print(f"Error loading test set: {e}")
        return

    all_preds = []
    all_labels = []
    all_probs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability of positive class (Fake) - assumes Fake is index 1? Check class_names

    # Check class index mapping
    # Usually sorted alphabetically: 'fake', 'real' -> fake=0, real=1? Or real/fake folders?
    # Let's inspect class_names
    print(f"Classes: {class_names}")
    fake_idx = class_names.index('fake') if 'fake' in class_names else 0 
    
    # Calculate Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, pos_label=fake_idx, average='binary')
    recall = recall_score(all_labels, all_preds, pos_label=fake_idx, average='binary')
    f1 = f1_score(all_labels, all_preds, pos_label=fake_idx, average='binary')

    print(f"\n--- Evaluation Report ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Fake): {precision:.4f}")
    print(f"Recall (Fake): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create reports directory first
    os.makedirs('ml/reports', exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('ml/reports/confusion_matrix.png')
    print("Saved confusion matrix to ml/reports/confusion_matrix.png")

    # ROC Curve
    # valid only if we have probs for the 'fake' class specifically
    # If fake is index 0 or 1, we need to be careful
    # all_probs calculated above assumed index 1.
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=fake_idx) 
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ml/reports/roc_curve.png')
    print("Saved ROC curve to ml/reports/roc_curve.png")

    os.makedirs('ml/reports', exist_ok=True)
    with open('ml/reports/evaluation_report.md', 'w') as f:
        f.write("# Model Evaluation Report\n")
        f.write(f"- Accuracy: {acc:.4f}\n")
        f.write(f"- Precision: {precision:.4f}\n")
        f.write(f"- Recall: {recall:.4f}\n")
        f.write(f"- F1 Score: {f1:.4f}\n")
        f.write(f"- AUC-ROC: {roc_auc:.4f}\n")

if __name__ == "__main__":
    evaluate_model()
