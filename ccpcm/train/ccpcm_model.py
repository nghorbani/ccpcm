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
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from configer import Configer
from ccpcm.tools.omni_tools import makepath, log2file
from torch import nn, optim
from torch.utils.data import DataLoader

from ccpcm.data.dataloader import CCPCM_DS
from ccpcm.tools.training_tools import EarlyStopping

class CCPCM(nn.Module):

    def __init__(self, window_size=10, n_class=40, n_neurons=256, **kwargs):
        super(CCPCM, self).__init__()

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.ll = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        H = 20
        L = 1
        self.bit_conv1 = nn.Conv1d(window_size, H, 1, stride=1)#alternatively one might want o learn temporal dynamics instead of product dynamics
        self.bit_dense1 = nn.Linear(n_class*H, 6, bias=False)

        self.dipt_dense1 = nn.Linear(n_class, L, bias=False)
        self.binf_dense1 = nn.Linear(n_class, L, bias=False)
        self.ztp1_dense1 = nn.Linear(n_class* (2*H+4), n_class, bias=False) # paper mentions 2H+5 but didnt understood why. maybe some bias?

    def forward(self, Ditp1, Bit, Binf):
        '''
        Given coupon assignment, customer purchasing history, and product purchasing frequencies, will predict pobability of customer buying product j
        :param
        :return:
        '''


        bs, window_size, J = Bit.shape
        bh_out = self.ll(self.bit_conv1(Bit)).transpose(1,2).contiguous()

        bh_bar_out = self.ll(self.bit_dense1(bh_out.view(bs, -1)))
        bh_bar_out = self.ll(torch.matmul(self.bit_dense1.weight.transpose(0,1), bh_bar_out.transpose(0,1)).transpose(0,1)).view(bs, J, -1)
        # bit_out = self.bit_conv2(bit_out.view(bs, -1, bit_H)).transpose(1,2) #should add ll?

        dipt1_out = self.ll(self.dipt_dense1(Ditp1.view(bs, -1)))
        dipt1_out = self.sigmoid(torch.matmul(self.dipt_dense1.weight.transpose(0,1), dipt1_out.transpose(0,1))).transpose(0,1).view(bs, J, 1)

        binf_out = self.ll(self.binf_dense1(Binf.view(bs, -1)))
        binf_out = self.sigmoid(torch.matmul(self.binf_dense1.weight.transpose(0,1), binf_out.transpose(0,1))).transpose(0,1).view(bs, J, 1)

        Z = torch.cat([bh_out, bh_bar_out,Ditp1.view([bs,J,1]), dipt1_out,Binf.view(bs, J, 1), binf_out  ], dim=-1)
        Ztp1 = self.ztp1_dense1(Z.view(bs,-1))#.view(bs, -1)

        pvalue = self.sigmoid(Ztp1)
        # pvalue = self.softmax(Ztp1) # Paper says softmax, however the test set the probability over products in a basket do not some to one. so either there is a category or this should be better sigmoid.

        result = {'Bit_p1': pvalue}

        return result


class CCPCM_Trainer:

    def __init__(self, work_dir, ps):

        from tensorboardX import SummaryWriter

        torch.manual_seed(ps.seed)

        starttime = datetime.now().replace(microsecond=0)
        ps.work_dir = makepath(work_dir, isfile=False)

        logger = log2file(makepath(os.path.join(work_dir, '%s.log' % (expr_code)), isfile=True))

        summary_logdir = os.path.join(work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training ccpcm experiment code %s' % (expr_code, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)

        logger('Base dataset_dir is %s' % ps.dataset_dir)

        shutil.copy2(os.path.basename(sys.argv[0]), work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda: torch.cuda.empty_cache()
        self.comp_device = torch.device("cuda:%d" % ps.cuda_id if torch.cuda.is_available() else "cpu")

        gpu_count = torch.cuda.device_count()
        logger('%d CUDAs available!' % gpu_count)

        gpu_brand = torch.cuda.get_device_name(ps.cuda_id) if use_cuda else None
        logger('Training with %s [%s]' % (self.comp_device, gpu_brand) if use_cuda else 'Training on CPU!!!')
        logger('Base dataset_dir is %s' % ps.dataset_dir)

        kwargs = {'num_workers': ps.n_workers}
        # kwargs = {'num_workers': ps.n_workers, 'pin_memory': True} if use_cuda else {'num_workers': ps.n_workers}
        ds_train = CCPCM_DS(dataset_dir=os.path.join(ps.dataset_dir, 'train'))
        self.ds_train = DataLoader(ds_train, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_val = CCPCM_DS(dataset_dir=os.path.join(ps.dataset_dir, 'vald'))
        self.ds_val = DataLoader(ds_val, batch_size=ps.batch_size, shuffle=True, drop_last=True, **kwargs)
        ds_test = CCPCM_DS(dataset_dir=os.path.join(ps.dataset_dir, 'test'))
        self.ds_test = DataLoader(ds_test, batch_size=ps.batch_size, shuffle=True, drop_last=False)

        logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f' %
               (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset)))

        self.ccpcm_model = CCPCM(window_size=ps.window_size, n_class=ps.n_class).to(self.comp_device)

        if ps.use_multigpu:
            self.ccpcm_model = nn.DataParallel(self.ccpcm_model)
            logger("Training on Multiple GPU's")

        varlist = [var[1] for var in self.ccpcm_model.named_parameters()]

        params_count = sum(p.numel() for p in varlist if p.requires_grad)
        logger('Total Trainable Parameters Count is %2.2f M.' % ((params_count) * 1e-6))

        self.optimizer = optim.Adam(varlist, lr=ps.base_lr, weight_decay=ps.reg_coef)

        self.logger = logger
        self.best_loss_total = np.inf
        self.try_num = ps.try_num
        self.epochs_completed = 0
        self.ps = ps

        if ps.best_model_fname is not None:
            self._get_model().load_state_dict(torch.load(ps.best_model_fname, map_location=self.comp_device), strict=False)
            logger('Restored model from %s' % ps.best_model_fname)

        self.BCELoss = nn.BCELoss()

    def _get_model(self):
        return self.ccpcm_model.module if isinstance(self.ccpcm_model, torch.nn.DataParallel) else self.ccpcm_model

    def train(self):
        self.ccpcm_model.train()
        save_every_it = len(self.ds_train) / self.ps.log_every_epoch
        train_loss_dict = {}
        for it, dorig in enumerate(self.ds_train):
            dorig = {k:dorig[k].to(self.comp_device) for k in dorig.keys()}

            self.optimizer.zero_grad()
            drec = self.ccpcm_model(Ditp1=dorig['Ditp1'], Bit=dorig['Bit'], Binf=dorig['Binf'])

            loss_total, cur_loss_dict = self.compute_loss(dorig, drec)
            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = CCPCM_Trainer.creat_loss_message(cur_train_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=it,
                                                              try_num=self.try_num, mode='train')

                self.logger(train_msg)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name='vald'):
        self.ccpcm_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                drec = self.ccpcm_model(Ditp1=dorig['Ditp1'], Bit=dorig['Bit'], Binf=dorig['Binf'])
                _, cur_loss_dict = self.compute_loss(dorig, drec)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):
        '''
        :param dorig: original data terms
        :param drec: reconstructed data
        :return:
        '''

        loss_dict = {
            'BCE': self.BCELoss(drec['Bit_p1'], dorig['Bit_p1']),
            }

        loss_dict['loss_total'] = torch.stack(list(loss_dict.values())).sum()

        return loss_dict['loss_total'], loss_dict

    def perform_training(self, num_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if num_epochs is None: num_epochs = self.ps.num_epochs

        self.logger(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), num_epochs))
        if message is not None: self.logger(expr_message)

        prev_lr = np.inf

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=7)
        early_stopping = EarlyStopping(patience=20)

        for epoch_num in range(1, num_epochs + 1):
            train_loss_dict = self.train()
            eval_loss_dict = self.evaluate()

            scheduler.step(eval_loss_dict['loss_total'])

            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                self.logger('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed += 1

            with torch.no_grad():
                eval_msg = CCPCM_Trainer.creat_loss_message(eval_loss_dict, expr_code=self.ps.expr_code,
                                                              epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                              try_num=self.try_num, mode='evald')
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.ps.best_model_fname = makepath(
                        os.path.join(self.ps.work_dir, 'snapshots', 'TR%02d_E%03d.pt' % (
                            self.try_num, self.epochs_completed)), isfile=True)
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.ccpcm_model.module.state_dict() if isinstance(self.ccpcm_model,
                                                                                     torch.nn.DataParallel) else self.ccpcm_model.state_dict(),
                               self.ps.best_model_fname)

                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss/scalars', {'train_loss_total': train_loss_dict['loss_total'],
                                                                'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

            if early_stopping(eval_loss_dict['loss_total']):
                self.logger("Early stopping")
                break

        endtime = datetime.now().replace(microsecond=0)
        self.logger(expr_message)
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        self.logger('Best model path: %s\n' % self.ps.best_model_fname)

    @staticmethod
    def creat_loss_message(loss_dict, expr_code='XX', epoch_num=0, it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s: [T:%.2e] - [%s]' % (
            expr_code, try_num, epoch_num, it, mode, loss_dict['loss_total'], ext_msg)


if __name__ == '__main__':

    expr_code = '10'

    default_ps_fname = 'ccpcm_defaults.ini'


    work_dir = os.path.join('../experiments/', expr_code)

    params = {
        'window_size': 10,
        'n_class': 40,
        'n_neurons': 256,
        'batch_size': 64, # each batch will be 120 frames
        'n_workers': 10,
        'cuda_id': 0,
        'use_multigpu':False,

        'reg_coef': 5e-4,

        'base_lr': 5e-3,

        'best_model_fname': None,
        'log_every_epoch': 2,
        'expr_code': expr_code,
        'work_dir': work_dir,
        'num_epochs': 100,
        'dataset_dir': '../data/dataset',
    }

    supercap_trainer = CCPCM_Trainer(work_dir, ps=Configer(default_ps_fname=default_ps_fname, **params))
    ps = supercap_trainer.ps

    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))

    expr_message = '\n[%s] batch_size=%d\n'% (ps.expr_code, ps.batch_size)

    expr_message += 'Using sigmoid instead of softmax.\n'
    expr_message += '\n'

    supercap_trainer.logger(expr_message)
    supercap_trainer.perform_training()
    ps.dump_settings(os.path.join(work_dir, 'TR%02d_%s' % (ps.try_num, os.path.basename(default_ps_fname))))