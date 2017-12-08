#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:42:59 2017

@author: anand
"""

from dataset import TextDataset

filename = "./books/book_EN_grimms_fairy_tails.txt"
data = TextDataset(filename)
d,_ = data.example(1)
#%%
batch = data.batch(10,20)

#%%
vocabsize = data.vocab_size

#%%
text = data.convert_to_string(d[0])