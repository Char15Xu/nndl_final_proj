import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()

    import torch, pandas as pd
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from clip_two_head_model import (CLIPTwoHeadClassifier,
                                     get_train_transforms,
                                     get_val_transforms)
    from clip_dataset import CLIPDataset

    scenarios = {
        "balanced":    ("balanced_train.csv",    "balanced_val.csv",    "model_balanced.pth"),
        "adversarial": ("adversarial_train.csv", "adversarial_val.csv", "model_adversarial.pth"),
        "train_like":  ("train_like_train.csv",  "train_like_val.csv",  "model_train_like.pth"),
        "novel":       ("novel_train.csv",       "novel_val.csv",       "model_novel.pth"),
    }

    # hard-coded count
    n_super = 4    # bird, dog, reptile, novel
    n_sub   = 88   # 87 seen + novel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()

    for name, (tr_csv, val_csv, out_pth) in scenarios.items():
        print(f"\n Scenario '{name}': train={tr_csv}, val={val_csv}")

        # datasets & loaders
        train_ds = CLIPDataset("train_images", tr_csv, n_subclasses=88, transform=get_train_transforms())
        val_ds   = CLIPDataset("train_images", val_csv, n_subclasses=88, transform=get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)

        # weight decay grouping
        decay, no_decay = [], []
        for nm, p in model.named_parameters():
            if not p.requires_grad: continue
            if nm.endswith(".bias") or "LayerNorm" in nm:
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = optim.AdamW([
            {'params': decay,    'weight_decay':1e-4},
            {'params': no_decay, 'weight_decay':0.0}
        ], lr=5e-4)

        # early stopping
        best_val = float('inf')
        wait = 0
        PATIENCE = 3
        EPOCHS = 10

        for ep in range(1, EPOCHS+1):
            # train
            model.train()
            total_loss = 0.0; cnt=0; cs=0; ct=0
            for imgs, sl, tl in train_loader:
                imgs, sl, tl = imgs.to(device), sl.to(device), tl.to(device)
                optimizer.zero_grad()
                ls, lt = model(imgs)
                loss = criterion(ls, sl) + criterion(lt, tl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                ps, pt = ls.argmax(1), lt.argmax(1)
                cs += (ps==sl).sum().item()
                ct += (pt==tl).sum().item()
                cnt += imgs.size(0)
            train_acc = (cs/cnt, ct/cnt)

            # validate
            model.eval()
            val_loss = 0.0; cnt=0; cs=0; ct=0
            with torch.no_grad():
                for imgs, sl, tl in val_loader:
                    imgs, sl, tl = imgs.to(device), sl.to(device), tl.to(device)
                    ls, lt = model(imgs)
                    val_loss += (criterion(ls, sl)+criterion(lt, tl)).item()
                    ps, pt = ls.argmax(1), lt.argmax(1)
                    cs += (ps==sl).sum().item()
                    ct += (pt==tl).sum().item()
                    cnt += imgs.size(0)
            val_acc = (cs/cnt, ct/cnt)

            print(f"Epoch {ep}/{EPOCHS}  "
                  f"TrainLoss {total_loss:.4f}  S/T {train_acc[0]:.3f}/{train_acc[1]:.3f}  "
                  f"ValLoss {val_loss:.4f}  S/T {val_acc[0]:.3f}/{val_acc[1]:.3f}")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(model.state_dict(), out_pth)
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"Early stopping at epoch {ep}")
                    break

        print(f"Saved best model to {out_pth}")
