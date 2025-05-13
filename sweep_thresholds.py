import torch
import pandas as pd
from torch.utils.data import DataLoader
from clip_two_head_model import CLIPTwoHeadClassifier, get_val_transforms
from clip_dataset import CLIPDataset
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
n_super, n_sub = 4, 88

# Load the novel‐scenario model
model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
model.load_state_dict(torch.load("model_novel.pth", map_location=device))
model.eval()

# Novel validation set
ds = CLIPDataset("train_images", "novel_val.csv", n_subclasses=n_sub, transform=get_val_transforms())
loader = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=True)

# Gather confidences and true labels
confs, labels = [], []
with torch.no_grad():
    for imgs, sl, _ in loader:
        imgs, sl = imgs.to(device), sl.to(device)
        ls, _ = model(imgs)
        p  = F.softmax(ls, dim=1)
        c,_ = p.max(dim=1)
        confs.append(c.cpu().numpy())
        labels.append(sl.cpu().numpy())
confs  = np.concatenate(confs)
labels = np.concatenate(labels)

novel_idx = n_super - 1
seen_mask   = labels != novel_idx
unseen_mask = labels == novel_idx

rows = []
for tau in np.linspace(0.0, 1.0, 101):
    # predict novel if confidence < tau
    preds = np.where(confs < tau, novel_idx, labels)

    seen_acc   = (preds[seen_mask]   == labels[seen_mask]  ).mean() if seen_mask.any() else None
    unseen_acc = (preds[unseen_mask] == labels[unseen_mask]).mean() if unseen_mask.any() else None

    rows.append({"tau": tau, "seen_super_acc": seen_acc, "unseen_super_acc": unseen_acc})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
