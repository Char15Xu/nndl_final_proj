import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, image_dir, csv_path, n_subclasses, transform=None):
        df = pd.read_csv(csv_path)

        # drop rows where image file doesn't exist or unreadable
        good = []
        for _, row in df.iterrows():
            path = os.path.join(image_dir, row['image'])
            if not os.path.isfile(path): 
                continue
            try:
                img = Image.open(path); img.verify()
                good.append(row)
            except (UnidentifiedImageError, OSError):
                continue

        self.df = pd.DataFrame(good).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.n_subclasses = n_subclasses
        self.novel_sub_idx = n_subclasses - 1  # last index is novel

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        s = int(row['superclass_index'])

        # map to novel_sub_idx if NaN
        raw = row.get('subclass_index', None)
        if pd.isna(raw):
            t = self.novel_sub_idx
        else:
            t = int(raw)

        return image, s, t
