import torch
import torch.nn as nn
import clip
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.75,1.33)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466,0.4578275,0.40821073),
            std=(0.26862954,0.26130258,0.27577711)
        ),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466,0.4578275,0.40821073),
            std=(0.26862954,0.26130258,0.27577711)
        ),
    ])

class CLIPTwoHeadClassifier(nn.Module):
    def __init__(self, backbone="ViT-B/32", n_super=4, n_sub=88, adapter_dim=512, dropout_p=0.3):
        super().__init__()
        self.clip_model, _ = clip.load(backbone, device=device, jit=False)
        # freeze backbone
        for p in self.clip_model.visual.parameters():
            p.requires_grad = False

        feat_dim = self.clip_model.visual.output_dim
        # small adapter + dropout
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
        )
        self.super_head = nn.Linear(adapter_dim, n_super)
        self.sub_head   = nn.Linear(adapter_dim, n_sub)

    def forward(self, x):
        feat = self.clip_model.encode_image(x)
        feat = feat.to(torch.float32)                 # for Linear
        h = self.adapter(feat)
        return self.super_head(h), self.sub_head(h)
