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
from tqdm import tqdm as tqdm
from ccpcm.tools.omni_tools import makepath
import os
import torch

data = pandas.read_csv(f"dataset/train.csv").sort_values(['i','t'])

customers = data['i'].to_numpy(dtype = np.int32)
times = data['t'].to_numpy(dtype = np.int32)
prods = data['j'].to_numpy(dtype = np.int32)
prices = data['price'].to_numpy(dtype = np.float32)
advertised = data['advertised'].to_numpy(dtype = np.bool)

n_cust = data.i.max() + 1
n_prod = data.j.max() + 1
T = data.t.max() + 1

B_path = makepath('dataset/train/B.npz', isfile=True)
if not os.path.exists(B_path):
    Bit = np.zeros([n_cust, T, n_prod], dtype=np.bool) # customer purchasing history: customer i purchased product j at time t

    for i in tqdm(range(n_cust)):
        for t in range(T):
            for j in range(n_prod):
                Bit[i, t, j] = np.sum(prods[(customers == i) & (times == t)] == j) > 0

    np.savez(B_path, data=Bit)
else:
    Bit = np.load(B_path)['data']

Binf = Bit.mean(1)#product purchasing frequencies

D_path = makepath('dataset/train/D.npz', isfile=True)
if not os.path.exists(D_path):
    Dit = np.zeros([n_cust, T, n_prod], dtype=np.float32) #coupon assignment: size of discount received by customer i in time t for product j

    price_orig = np.zeros([n_prod, 1])
    for j in range(n_prod):
        price_orig[j] = np.median(prices[(prods == j) & (advertised == False)])

    for i in tqdm(range(n_cust)):
        for t in range(T):
            for j in np.arange(n_prod)[Bit[i, t] == True]:
                price = prices[(customers == i) & (times == t) & (prods == j)]
                Dit[i, t, j] = (price_orig[j]-price)/price_orig[j]

    np.savez(D_path, data=Dit)
else:
    Dit = np.load(D_path)['data']

window_size = 10
ovsize = 9
time_ids = [np.arange(0, T)[i:i + window_size] for i in range(0, T, window_size - ovsize)]

data_Ditp1 = []
data_Bit = []
data_Binf = []
Bit_p1 = []
metas_i = []
metas_t = []

for tIds in time_ids:
    end_t = tIds[-1]
    if end_t == T - 1: continue
    for i in range(n_cust):
        data_Ditp1.append(Dit[i, end_t])
        data_Bit.append(Bit[i, tIds].astype(np.int32))
        data_Binf.append(Binf[i])
        Bit_p1.append(Bit[i, end_t+1].astype(np.int32))
        metas_i.append(i)
        metas_t.append(tIds)

data_train = {
    'Ditp1': np.stack(data_Ditp1),#Dit+1
    'Bit' : np.stack(data_Bit),
    'Binf' : np.stack(data_Binf),
    'Bit_p1' : np.stack(Bit_p1),#Bit+1
    'metas_i' : np.stack(metas_i),
    'metas_t' : np.stack(metas_t)
}

n_data = len(Bit_p1)
np.random.seed(100)
vald_ids = np.random.choice(n_data, int(n_data*0.1), replace=False)
train_ids = list(set(range(n_data)).difference(set(vald_ids)))

outpath = makepath(os.path.join('dataset/train/Bit_p1.pt'), isfile=True)
for k, v in data_train.items():
    torch.save(torch.tensor(v[train_ids]), outpath.replace('Bit_p1.pt', '%s.pt'%k))

outpath = makepath(os.path.join('dataset/vald/Bit_p1.pt'), isfile=True)
for k, v in data_train.items():
    torch.save(torch.tensor(v[vald_ids]), outpath.replace('Bit_p1.pt', '%s.pt'%k))

# Test will be applied on week 50th
# Note: testset specifies discount on product 24 however, it has never been discounted before in the trainset
prediction_example = pandas.read_csv("dataset/prediction_example.csv").sort_values(['i','j'])
promotion_schedule = pandas.read_csv("dataset/promotion_schedule.csv").sort_values(['j'])

test_Ditp1 = []
test_Bit = []
test_Binf = []
test_Bit_p1 = []
test_metas_i = []

cur_dipt1 = np.zeros(n_prod)
for j in range(n_prod):
    discount = promotion_schedule.query('j == %d' % j).discount.to_numpy()
    if len(discount) > 0: cur_dipt1[j] = discount

T_test = 49
tIds = range(T_test-window_size, T_test)
for i in list(set(prediction_example.i)):

    test_Ditp1.append(cur_dipt1)
    test_Bit.append(Bit[i, tIds].astype(np.int32))
    test_Binf.append(Binf[i])
    test_Bit_p1.append(prediction_example.query('i==%d'%i).prediction.to_numpy())
    test_metas_i.append(i)

data_test = {
    'Ditp1': np.stack(test_Ditp1),#Dit+1
    'Bit' : np.stack(test_Bit),
    'Binf' : np.stack(test_Binf),
    'Bit_p1' : np.stack(test_Bit_p1),#Bit+1
    'metas_i' : np.stack(test_metas_i),
}

outpath = makepath(os.path.join('dataset/test/Bit_p1.pt'), isfile=True)
for k, v in data_test.items():
    torch.save(torch.tensor(v), outpath.replace('Bit_p1.pt', '%s.pt'%k))
