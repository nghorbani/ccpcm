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
import os, glob


def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]

    print(('Found CCPCM Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate ccpcm_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_ccpcm(expr_dir, ccpcm_model='snapshot'):
    '''

    :param expr_dir:
    :param ccpcm_model: either 'snapshot' to use the experiment folder's code or a ccpcm imported module, e.g.
    from ccpcm.train.ccpcm_smpl import ccpcm, then pass ccpcm to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if ccpcm_model == 'snapshot':

        ccpcm_path = sorted(glob.glob(os.path.join(expr_dir, 'ccpcm_*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('ccpcm', ccpcm_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ccpcm_pt = getattr(module, 'CCPCM')(**ps)
    else:
        ccpcm_pt = ccpcm_model(**ps)

    ccpcm_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    ccpcm_pt.eval()

    return ccpcm_pt, ps


if __name__ == '__main__':
    expr_dir = '../experiments/10'
    ccpcm_pt, ps = load_ccpcm(expr_dir, ccpcm_model='snapshot')
