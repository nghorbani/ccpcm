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
def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def log2file(logpath=None, auto_newline = True):
    import sys
    if logpath is not None:
        makepath(logpath, isfile=True)
        fhandle = open(logpath,'a+')
    else:
        fhandle = None
    def _(text):
        if text is None: return
        if auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        sys.stderr.write(text)
        if fhandle is not None:
            fhandle.write(text)
            fhandle.flush()

    return lambda text: _(text)

copy2cpu = lambda tensor: tensor.detach().cpu().numpy()
