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
class BaseMultiTask(NN):
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
  '''
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject):
    """"""

    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 1:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = vocabs[1][token[2]]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          #gold_parse = gold[1]
          gold_parse = gold[0]
          #gold_rel = vocabs[2][gold[2]]
          gold_rel = vocabs[2][gold[1]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return
  '''
  #=============================================================
  def validate(self, mb_inputs, mb_targets, mb_probs, sequence):
    """"""

    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    for inputs, targets, parse_probs, rel_probs, srl_pred in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs, sequence):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      # Filter Root here
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)

      #sent = -np.ones( (length, 10), dtype=int)
      sent = -np.ones((length, 3), dtype=int)
      tokens = np.arange(1, length+1)
      sent[:, 0] = parse_preds[tokens]
      sent[:, 1] = rel_preds[tokens]
      sent[:, 2] = srl_pred[tokens]

      # sent[:,0] = tokens
      # sent[:,1:4] = inputs[tokens,0:3]
      # #sent[:,4] = targets[tokens,0]
      # #sent[:,4] = inputs[tokens, 2]
      # sent[:,4] = parse_preds[tokens]
      # sent[:,5] = rel_preds[tokens]
      # sent[:,6:-1] = targets[tokens,:]
      # sent[:,-1] = srl_pred[tokens]
      sents.append(sent)
    return sents

  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2, 6, 7)
  @property
  def target_idxs(self):
    return (3, 4, 5)
