# NNDL Final Project
#### Team members: Charles Xu, Zixiang Qu, Lingyi Zhang

This repo contains all necessary code to reproduce the result in the report. 
Please download all necessary data using the Google drive link in the report.

There are two implementations for our project:
## 1. CLIP with Two Classification Heads -- Zixiang Qu
## Environment Setup

Install dependencies (Python 3.8+):

```bash
pip install torch torchvision pandas tqdm ftfy regex sentencepiece clip
```

## Usage Instructions

Please remember to put the dataset files that inside our Google Drive link into the same directory as the python files in this GitHub repository.

### 1. Split-Based Experiments

```bash
# Train on four validation splits with early stopping:
python train_clip_two_head.py

# Evaluate metrics on balanced, adversarial, train-like, and novel splits:
python evaluate_splits.py
```

### 2. Final Full-Data Training & Baseline Submission

```bash
# Train on the entire training set (no splits):
python train_full_clip.py

# Generate baseline submission (no tuned threshold):
python predict_full_clip.py
```

### 3. Tuning Novel Detection

1. Sweep confidence threshold τ on the novel validation set:

   ```bash
   python sweep_thresholds.py
   ```
2. Edit `predict_with_threshold.py`, set `TAU_HEAD` to the chosen value from the sweep.
3. Generate final submission with tuned threshold and zero-shot fallback:

   ```bash
   python predict_with_threshold.py
   ```

## Submission File

* Final predictions are saved as `clip_submission.csv` with header:

  ```csv
  image,superclass_index,subclass_index
  ```



## 2. ConvNeXT + CLIP implementation -- Charles Xu
Please move the dataset folder to your Google Drive and run all cells in `convnext_clip.ipynb` to reproduce the results.
