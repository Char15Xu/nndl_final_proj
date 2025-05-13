import os, torch, pandas as pd
from PIL import Image
from tqdm import tqdm
import clip
import torch.nn.functional as F

from clip_two_head_model import CLIPTwoHeadClassifier, get_val_transforms

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tuned thresholds
    TAU_HEAD = 0.98    # get by running sweep_threshold.py
    TAU_CLIP = 0.20

    # Load CLIP zero-shot for fallback
    clip_model, clip_pre = clip.load("ViT-B/32", device=device, jit=False)
    super_labels = ["bird","dog","reptile"]
    text_prompts = [f"a photo of a {lbl}" for lbl in super_labels]
    text_tokens  = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    # Load my full-data finetuned head
    n_super, n_sub = 4, 88
    model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
    model.load_state_dict(torch.load("model_full.pth", map_location=device))
    model.eval()

    prep = get_val_transforms()
    TEST_DIR = "test_images"
    OUT_CSV  = "clip_submission.csv"
    rows = []

    for fn in tqdm(sorted(os.listdir(TEST_DIR), key=lambda x: int(os.path.splitext(x)[0]))):
        img = Image.open(os.path.join(TEST_DIR,fn)).convert("RGB")
        x = prep(img).unsqueeze(0).to(device)

        with torch.no_grad():
            ls, lt = model(x)
            probs_super = F.softmax(ls, dim=1)
            conf, pred = probs_super.max(dim=1)
            conf = conf.item()
            pred = pred.item()

        if conf < TAU_HEAD:
            pred = n_super - 1

        if pred != n_super-1:
            with torch.no_grad():
                img_feat = clip_model.encode_image(clip_pre(img).unsqueeze(0).to(device))
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims = (text_feats @ img_feat.T).squeeze(1)
            if sims.max().item() < TAU_CLIP:
                pred = n_super - 1

        sub = lt.argmax(dim=1).item()

        rows.append({
            'image': fn,
            'superclass_index': pred,
            'subclass_index':   sub
        })

    pd.DataFrame(rows, columns=['image','superclass_index','subclass_index'])\
      .to_csv(OUT_CSV, index=False)

    print("Wrote", OUT_CSV)
