# Cross-Category Product Choice Model
A PyTorch implementation for the paper 
[Cross-Category Product Choice: A Scalable Deep-Learning Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3402471).

The test set error is too high. This could have multiple reasons:
* Overfitting to training data
* Network has failed to learn anything useful from the training data that is applicable to test data
* Train data has nothing to do with the test data

Since we have implemented a network that has been tested in a publication before, there is more reason to believe that the issue lies with the
 training data rather than the method.
This hypothesis can be tested for example by customer 1. In the test data customer 1 is predicted to purchase item 38 
with probability of 0.91, however the purchase frequency of item 38 from all the T of customer 1 is exactly zero, e.g. the customer never bought it before.
Since this item is also not discounted we see evidence in favor of inefficiency of the training data for the current test data. 
However, bugs in the newtork architecture are also plasubile, and should be addressed after the aformentioned issue is addressed.
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
------- VALD ----------
vald dataset size: 7800
BCE = 8.37e-02
------- TEST ----------
test dataset size: 3
BCE = 2.37e+00
```

