# Cross-Category Product Choice Model
A PyTorch implementation for the paper 
[Cross-Category Product Choice: A Scalable Deep-Learning Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3402471).

# Install
```bash
pip install git+https://github.com/nghorbani/ccpcm
```
or inside the ccpcm root directory run:
```bash
python setup.py develop build
```
# Train:
First you need to prepare the training data by running the script ccpcm/data/prepare_data.py.
It assumes that the data containing files *prediction_example.csv, promotion_schedule.csv, train.csv* are present in the ccpcm/data/dataset folder.
The script will output the necessary PyTorch *.pt* files in the ccpcm/data/dataset directory.
Edit the ccpcm/train/ccpcm_model.py to point to the right dataset dir and change the hyperparameters to your desire.
Afterward you can run the training code from the *ccpcm/train* directory to make the experiment:
```bash
python -m ccpcm_model.py
``` 

# Evaluate:
Given that you have the experiments folder inside the ccmpm directory, simply edit *ccpcm/eval/eval/ccpcm.py* 
to point to the experiemnt id that you want to evaluate and run the following command from eval directory, e.g. *ccpcm/eval/eval/*:
```bash
python -m eval_ccpcm
``` 

```
Found CCPCM Trained Model: ../experiments/10/snapshots/TR00_E057.pt
Model Found: [../experiments/10/snapshots/TR00_E057.pt] [Running on dataset: ../data/dataset] 
------- TRAIN ----------
train dataset size: 70200
BCE = 8.37e-02
AUC: 0: 0.55, 1: 0.62, 2: 0.59, 3: 0.59, 4: 0.54, 5: 0.68, 6: 0.64, 7: 0.60, 8: 0.53, 9: 0.53, 
10: 0.53, 11: 0.53, 12: 0.51, 13: 0.53, 14: 0.53, 15: 0.63, 16: 0.57, 17: 0.53, 18: 0.49, 19: 0.47,
 20: 0.50, 21: 0.54, 22: 0.54, 23: 0.52, 24: 0.56, 25: 0.54, 26: 0.53, 27: 0.55, 28: 0.52, 29: 0.49, 
 30: 0.70, 31: 0.47, 32: 0.51, 33: 0.68, 34: 0.61, 35: 0.53, 36: 0.55, 37: 0.50, 38: 0.49, 39: 0.72
------- VALD ----------
vald dataset size: 7800
BCE = 8.37e-02
AUC: 0: 0.61, 1: 0.63, 2: 0.60, 3: 0.61, 4: 0.41, 5: 0.68, 6: 0.65, 7: 0.54, 8: 0.53, 9: 0.47, 10: 0.56,
 11: 0.52, 12: 0.55, 13: 0.57, 14: 0.54, 15: 0.64, 16: 0.58, 17: 0.57, 18: 0.56, 19: 0.39, 20: 0.31, 
 21: 0.56, 22: 0.48, 23: 0.68, 24: 0.56, 25: 0.47, 26: 0.50, 27: 0.44, 28: 0.53, 29: 0.47, 30: 0.70, 
 31: 0.53, 32: 0.48, 33: 0.68, 34: 0.64, 35: 0.48, 36: 0.50, 37: 0.51, 38: 0.42, 39: 0.71
```

