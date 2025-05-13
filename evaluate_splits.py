import torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip_two_head_model import CLIPTwoHeadClassifier, get_val_transforms
from clip_dataset import CLIPDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

splits = {
    "balanced":    ("balanced_val.csv",    "model_balanced.pth"),
    "adversarial": ("adversarial_val.csv", "model_adversarial.pth"),
    "train_like":  ("train_like_val.csv",  "model_train_like.pth"),
    "novel":       ("novel_val.csv",       "model_novel.pth"),
}

n_super, n_sub = 4, 88
results = []

for name, (val_csv, ckpt) in splits.items():
    # load model
    model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # data
    ds = CLIPDataset("train_images", val_csv, n_subclasses=n_sub, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, num_workers=0)

    total = 0
    corr_s = corr_t = 0
    loss_s = loss_t = 0.0
    seen_s = seen_t = seen_cnt = 0
    nov_s = nov_t = nov_cnt = 0

    with torch.no_grad():
        for imgs, sl, tl in loader:
            imgs, sl, tl = imgs.to(device), sl.to(device), tl.to(device)
            ls, lt = model(imgs)
            ps = ls.argmax(1)
            pt = lt.argmax(1)

            # overall accuracy
            corr_s += (ps==sl).sum().item()
            corr_t += (pt==tl).sum().item()
            total += imgs.size(0)

            # loss
            loss_s += F.cross_entropy(ls, sl, reduction="sum").item()
            loss_t += F.cross_entropy(lt, tl, reduction="sum").item()

            # seen vs novel
            novel_s_idx = n_super-1
            mask_seen = sl != novel_s_idx
            seen_cnt += mask_seen.sum().item()
            nov_cnt  += (~mask_seen).sum().item()
            seen_s += (ps[mask_seen]==sl[mask_seen]).sum().item()
            nov_s  += (ps[~mask_seen]==sl[~mask_seen]).sum().item()

    results.append({
        "split": name,
        "super_acc": corr_s/total,
        "sub_acc":   corr_t/total,
        "super_loss": loss_s/total,
        "sub_loss":   loss_t/total,
        "super_seen_acc": seen_s/seen_cnt if seen_cnt else None,
        "super_nov_acc":  nov_s/nov_cnt   if nov_cnt  else None,
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

