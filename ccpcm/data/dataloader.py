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

import os
import torch
from torch.utils.data import Dataset
import glob

class CCPCM_DS(Dataset):
    def __init__(self, dataset_dir):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
       return len(self.ds['Bit'])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        return {k: self.ds[k][idx].type(torch.float32) for k in self.ds.keys()}

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    batch_size = 256
    dataset_dir = '../data/dataset/test'

    ds = CCPCM_DS(dataset_dir=dataset_dir)
    print('dataset size: %d'%len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        print([(k,v.shape) for k,v in sample_batched.items()])
        break
