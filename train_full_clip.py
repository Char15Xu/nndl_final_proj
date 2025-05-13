import torch, pandas as pd, multiprocessing
from torch import nn, optim
from torch.utils.data import DataLoader
from clip_two_head_model import CLIPTwoHeadClassifier, get_train_transforms
from clip_dataset import CLIPDataset

if __name__=="__main__":
    multiprocessing.freeze_support()

    TRAIN_CSV = "train_data.csv"
    IMAGE_DIR = "train_images"
    MODEL_OUT = "model_full.pth"

    n_super = 4
    n_sub   = 88

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH, EPOCHS, LR = 32, 10, 5e-4
    criterion = nn.CrossEntropyLoss()

    # Dataset & Loader with full data
    train_ds = CLIPDataset(
        image_dir=IMAGE_DIR,
        csv_path=TRAIN_CSV,
        n_subclasses=n_sub,
        transform=get_train_transforms()
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=0, pin_memory=True
    )

    # Model, Optimizer & Weight Decay Grouping
    model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.endswith(".bias") or "LayerNorm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = optim.AdamW([
        {'params': decay,    'weight_decay':1e-4},
        {'params': no_decay, 'weight_decay':0.0}
    ], lr=LR)

    # Training Loop
    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        correct_s = correct_t = total = 0

        for imgs, sl, tl in train_loader:
            imgs, sl, tl = imgs.to(device), sl.to(device), tl.to(device)
            optimizer.zero_grad()
            ls, lt = model(imgs)
            loss = criterion(ls, sl) + criterion(lt, tl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ps, pt = ls.argmax(1), lt.argmax(1)
            correct_s += (ps==sl).sum().item()
            correct_t += (pt==tl).sum().item()
            total += imgs.size(0)

        print(f"Epoch {ep}/{EPOCHS}  Loss {total_loss:.4f}  "
              f"SuperAcc {correct_s/total:.3f}  SubAcc {correct_t/total:.3f}")


    torch.save(model.state_dict(), MODEL_OUT)
    print("Saved full-model checkpoint →", MODEL_OUT)
