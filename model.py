import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class AffineFusion(nn.Module):
    def __init__(self, d_extra=768, d_img=512):
        super().__init__()
        self.generator = nn.Sequential(
            nn.LayerNorm(d_extra),
            nn.Linear(d_extra, 2 * d_img)
        )

    def forward(self, x, extra_feat):
        gamma, beta = self.generator(extra_feat).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class MutiModelAF(nn.Module):
    def __init__(self, biobert_model_name, num_classes=2):
        super().__init__()

        self.biobert = AutoModel.from_pretrained(biobert_model_name)
        
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.film = AffineFusion(d_extra=768, d_img=512)
        self.residual_block = ResidualBlock(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images, tokenized_text):

        bert_outputs = self.biobert(**tokenized_text)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :] # [B, 768]

        img_feat = self.img_encoder(images)
        img_feat = self.global_avg_pool(img_feat) # [B, 512, 1, 1]

        x = self.film(img_feat, cls_embedding)
        x = self.residual_block(x)
        return self.classifier(x.flatten(1))