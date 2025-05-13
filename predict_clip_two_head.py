import os, torch, pandas as pd
from PIL import Image
from tqdm import tqdm

from clip_two_head_model import CLIPTwoHeadClassifier, get_val_transforms

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ensemble these four models
    ckpts = [
        "model_balanced.pth",
        "model_adversarial.pth",
        "model_train_like.pth",
        "model_novel.pth"
    ]

    n_super, n_sub = 4, 88

    # load models
    models = []
    for pth in ckpts:
        m = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
        m.load_state_dict(torch.load(pth, map_location=device))
        m.eval()
        models.append(m)

    prep = get_val_transforms()
    TEST_DIR = "test_images"
    OUT_CSV  = "clip_submission.csv"
    rows = []

    # natural numeric sort
    files = sorted(os.listdir(TEST_DIR), key=lambda fn: int(os.path.splitext(fn)[0]))

    for fn in tqdm(files):
        img = Image.open(os.path.join(TEST_DIR,fn)).convert("RGB")
        x = prep(img).unsqueeze(0).to(device)

        # sum logits from all models
        sum_ls = sum(m(x)[0] for m in models)
        sum_lt = sum(m(x)[1] for m in models)
        ps = int(sum_ls.argmax(1))
        pt = int(sum_lt.argmax(1))

        rows.append({'image':fn,'superclass_index':ps,'subclass_index':pt})

    pd.DataFrame(rows, columns=['image','superclass_index','subclass_index'])\
      .to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)
