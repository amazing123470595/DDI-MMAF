import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def evaluate(model, dataloader, device, criterion, tokenizer, cell_to_text_dict):
    model.eval()
    val_labels, val_preds, val_probs = [], [], []
    val_running_loss = 0.0

    with torch.no_grad():
        for images, cell_names, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            texts = [f"{name} is a {cell_to_text_dict.get(name, 'unknown')} cancer cell line." for name in cell_names]
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)

            outputs = model(images, inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_metrics = {
        'val_loss': val_running_loss / len(dataloader),
        'precision': precision_score(val_labels, val_preds, zero_division=0),
        'recall': recall_score(val_labels, val_preds, zero_division=0),
        'f1': f1_score(val_labels, val_preds, zero_division=0),
        'balanced_acc': balanced_accuracy_score(val_labels, val_preds),
        'mcc': matthews_corrcoef(val_labels, val_preds),
        'roc_auc': roc_auc_score(val_labels, val_probs)
    }
    return val_metrics

def train_model(model, train_loader, test_loader, device, criterion, optimizer, 
                tokenizer_name, cell_to_text_dict, num_epochs=100):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    best_bal_acc = 0.0
    early_stop = 0
    patience = 20

    best_metrics = {
        "epoch": 0,
        "balanced_acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "mcc": 0.0
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, cell_names, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                images, labels = images.to(device), labels.to(device)

                texts = [f"{name} is a {cell_to_text_dict.get(name, 'unknown')} cancer cell line." for name in cell_names]
                inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)

                optimizer.zero_grad()
                outputs = model(images, inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        train_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, test_loader, device, criterion, tokenizer, cell_to_text_dict)


        print(f"Epoch: {epoch + 1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Balanced Acc: {val_metrics['balanced_acc']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"ROC AUC: {val_metrics['roc_auc']:.4f} | "
              f"MCC: {val_metrics['mcc']:.4f}")

        if val_metrics['balanced_acc'] > best_bal_acc:
            best_bal_acc = val_metrics['balanced_acc']
            save_model(model, 'checkpoints/best_model.pth')
            print(f"Best model saved! (Balanced Accuracy: {best_bal_acc:.4f})")

            best_metrics.update({
                "epoch": epoch + 1,
                "balanced_acc": val_metrics['balanced_acc'],
                "precision": val_metrics['precision'],
                "recall": val_metrics['recall'],
                "f1": val_metrics['f1'],
                "roc_auc": val_metrics['roc_auc'],
                "mcc": val_metrics['mcc']
            })
            early_stop = 0
        else:
            early_stop += 1

        if early_stop > patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    with open("best_metrics.txt", "a") as f:
        f.write("\nBest Validation Metrics\n")
        f.write("=======================\n")
        metric_str = " | ".join([f"{k}: {v}" for k, v in best_metrics.items()])
        f.write(metric_str + "\n")
    
    print(f"Best metrics saved to best_metrics.txt")