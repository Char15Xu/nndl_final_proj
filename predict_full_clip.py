import os, torch, pandas as pd
from PIL import Image
from tqdm import tqdm

from clip_two_head_model import CLIPTwoHeadClassifier, get_val_transforms

if __name__=="__main__":
    TEST_DIR = "test_images"
    MODEL_PTH= "model_full.pth"
    OUT_CSV  = "clip_submission.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_super, n_sub = 4, 88

    # load model
    model = CLIPTwoHeadClassifier(n_super=n_super, n_sub=n_sub).to(device)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.eval()

    prep = get_val_transforms()
    rows = []

    # numeric sort filenames
    files = sorted(os.listdir(TEST_DIR), key=lambda fn: int(os.path.splitext(fn)[0]))

    for fn in tqdm(files):
        img = Image.open(os.path.join(TEST_DIR,fn)).convert("RGB")
        x = prep(img).unsqueeze(0).to(device)
        with torch.no_grad():
            ls, lt = model(x)
        ps, pt = int(ls.argmax(1)), int(lt.argmax(1))
        rows.append({'image':fn,'superclass_index':ps,'subclass_index':pt})

    pd.DataFrame(rows, columns=['image','superclass_index','subclass_index']).to_csv(OUT_CSV, index=False)
    print("Wrote submission →", OUT_CSV)
