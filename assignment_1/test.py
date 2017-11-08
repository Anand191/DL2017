#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:27:42 2017

@author: anand
"""

import numpy as np
import cifar10_utils
import os

#%%
dir1 = os.getcwd()

cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')

#%%
batch_size=10
x, y = cifar10.train.next_batch(batch_size)

xt, yt = cifar10.test.images, cifar10.test.labels