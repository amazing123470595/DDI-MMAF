import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_file']
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]['label'])
        cell_name = self.data.iloc[idx]["cell"]

        if self.transform:
            image = self.transform(image)

        return image, cell_name, label

default_transform = transforms.Compose([
    transforms.Resize((224, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
