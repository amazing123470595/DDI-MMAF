import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AdamW
from dataloader import CustomImageDataset, default_transform
from model import MutiModelAF
from train import train_model
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data loader
    train_dataset = CustomImageDataset(config.train_csv, transform=default_transform)
    test_dataset = CustomImageDataset(config.test_csv, transform=default_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load the text mapping table from Cell to Tissue
    mapping_df = pd.read_csv("dataset/cell_tissue.csv")
    cell_to_text_dict = dict(zip(mapping_df['cell_line'], mapping_df['tissue']))

    model = MutiModelAF("dmis-lab/biobert-v1.1", config.num_classes).to(device)

    # Set a variable learning rate
    optimizer = AdamW([
        {'params': model.biobert.parameters(), 'lr': 2e-5},
        {'params': model.img_encoder.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss().to(device)

    train_model(
        model, train_loader, test_loader, device, criterion, optimizer,
        tokenizer_name="dmis-lab/biobert-v1.1",
        cell_to_text_dict=cell_to_text_dict,
        num_epochs=config.num_epochs
    )

if __name__ == "__main__":
    main()