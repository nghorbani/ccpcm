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
import json

from ccpcm.data.dataloader import CCPCM_DS

from ccpcm.tools.omni_tools import makepath
from ccpcm.tools.omni_tools import copy2cpu as c2c

from torch.utils.data import DataLoader
import torch

def evaluate_error(dataset_dir, ccpcm_model, ccpcm_ps, splitname, batch_size=1):
    ccpcm_model.eval()

    ds_name = dataset_dir.split('/')[-2]

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ccpcm_model = ccpcm_model.to(comp_device)

    BCELoss = torch.nn.BCELoss()

    ds = CCPCM_DS(dataset_dir=os.path.join(dataset_dir, splitname))
    print('%s dataset size: %s'%(splitname,len(ds)))
    ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)#batchsize for bm is fixed so drop the last one

    loss_mean = []
    with torch.no_grad():
        for dorig in ds:
            dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

            drec = ccpcm_model(Ditp1=dorig['Ditp1'], Bit=dorig['Bit'], Binf=dorig['Binf'])
            loss_mean.append(BCELoss(drec['Bit_p1'], dorig['Bit_p1']))

    final_results = {
        'BCE': float(c2c(torch.stack(loss_mean).mean())),
    }

    outpath = makepath(os.path.join(ccpcm_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(ccpcm_ps.best_model_fname).replace('.pt','_%s.json'%splitname)), isfile=True)
    with open(outpath, 'w') as f:
        json.dump(final_results,f)

    return final_results

if __name__ == '__main__':
    from ccpcm.tools.model_loader import load_ccpcm

    expr_code = '10'

    expr_dir = os.path.join('../experiments/', expr_code)

    ccpcm_model, ccpcm_ps = load_ccpcm(expr_dir)
    dataset_dir = ccpcm_ps.dataset_dir
    print('Model Found: [%s] [Running on dataset: %s] '%(ccpcm_ps.best_model_fname, dataset_dir))
    for splitname in ['train', 'vald','test']:
        print('------- %s ----------'%splitname.upper())
        results = evaluate_error(dataset_dir, ccpcm_model, ccpcm_ps, splitname, batch_size=512)
        print('BCE = %.2e' % (results['BCE']))

