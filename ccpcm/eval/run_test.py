# -*- coding: utf-8 -*-
# Part of [CCPCM Software](https://github.com/nghorbani/ccpcm)
# See LICENSE file for full copyright and licensing details.
# Implementation: Nima Ghorbani: nghorbani.github.io
#
# If you use this code please consider citing:
# Cross-Category Product Choice: A Scalable Deep-Learning Model (Sebastian Gabel and Artem Timoshenko)
#
#
# 2019.09.01
import pandas
import numpy as np
from ccpcm.tools.model_loader import load_ccpcm
from ccpcm.tools.omni_tools import makepath
from ccpcm.tools.omni_tools import copy2cpu as c2c
import os
import torch

B_path = makepath('../data/dataset/train/B.npz', isfile=True)
Bit = np.load(B_path)['data']
Binf = Bit.mean(1)  # product purchasing frequencies
window_size = 10
n_prod = 40

expr_code = '10'
expr_dir = os.path.join('../experiments/', expr_code)
ccpcm_model, ccpcm_ps = load_ccpcm(expr_dir)

# Test will be applied on week 50th
# Note: testset specifies discount on product 24 however, it has never been discounted before in the trainset
prediction_example = pandas.read_csv("../data/dataset/prediction_example.csv")  # .sort_values(['i','j'])
promotion_schedule = pandas.read_csv("../data/dataset/promotion_schedule.csv")  # .sort_values(['j'])
prediction_results = pandas.DataFrame(columns=list(prediction_example.columns))

dipt1 = np.zeros(n_prod)
for j in range(n_prod):
    discount = promotion_schedule.query('j == %d' % j).discount.to_numpy()
    if len(discount) > 0: dipt1[j] = discount

T_test = 49
tIds = range(T_test - window_size, T_test)
for d in prediction_example.iterrows():
    i, j, gt_prediction = int(d[1].i), int(d[1].j), d[1].prediction
    prediction = c2c(ccpcm_model(
        Ditp1=torch.tensor(dipt1[np.newaxis], dtype=torch.float32),
        Bit=torch.tensor(Bit[i, tIds][np.newaxis], dtype=torch.float32),
        Binf=torch.tensor(Binf[i][np.newaxis], dtype=torch.float32))['Bit_p1'])[0, j]
    print(i, j, gt_prediction, prediction)

    prediction_results = prediction_results.append({'i': i, 'j': j, 'prediction': prediction}, ignore_index=True)

prediction_results.to_csv("../data/dataset/prediction_results.csv", index=False)
