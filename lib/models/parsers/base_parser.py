#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    raise NotImplementedError

  #=============================================================
  def validate(self, mb_inputs, mb_targets, mb_probs):
    """"""

    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      # Filter Root here
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)

      #sent = -np.ones( (length, 9), dtype=int)
      sent = -np.ones((length,2), dtype=int)
      tokens = np.arange(1, length+1)
      sent[:,0] = parse_preds[tokens]
      sent[:,1] = rel_preds[tokens]
      # sent[:,0] = tokens
      # sent[:,1:4] = inputs[tokens] # Here we filter root because token start from 1
      # sent[:,4] = targets[tokens,0]
      # sent[:,5] = parse_preds[tokens]
      # sent[:,6] = rel_preds[tokens]
      # sent[:,7:] = targets[tokens, 1:]
      sents.append(sent)
    return sents

  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2)
  @property
  def target_idxs(self):
    return (3, 4)
