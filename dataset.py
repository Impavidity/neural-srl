#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter

from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket

#***************************************************************
class Dataset(Configurable):
  """"""

  #=============================================================
  def __init__(self, filename, vocabs, builder, *args, **kwargs):
    """"""

    super(Dataset, self).__init__(*args, **kwargs)
    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.vocabs = vocabs
    self.rebucket()

    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    if self.model_type == "SimpleSrler" or self.model_type == "SenseDisamb":
      self.targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='targets')
    else:
      print("Unsupported Mode in target placeholder")
      exit()
    self.builder = builder() # It is not be used

  #=============================================================
  def file_iterator(self, filename):
    """"""

    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line:
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if buff[-1]:
                  buff.append([])
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line:
            buff[-1].append(line)
          else:
            if buff[-1]:
              buff.append([])
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff

  #=============================================================
  def _process_buff(self, buff):
    """"""
    #Todo: Change process_buff for input\
    if self.model_type == "SimpleSrler":
      words, poss, srls, verbs, is_verbs, verb_senses = self.vocabs
      for i, sent in enumerate(buff):
        is_verbs_index = [1 if item[is_verbs.conll_idx] == '1' else 0 for item in sent]
        for item in range(len(is_verbs_index)):
          if is_verbs_index[item] == 1:
            continue
          else:
            sent[item][verbs.conll_idx] = "<PAD>"
        for j, token in enumerate(sent):
          word, pos, verb, is_verb, srl, verb_sense = token[words.conll_idx], token[poss.conll_idx], token[verbs.conll_idx], token[is_verbs.conll_idx], token[srls.conll_idx], token[verb_senses.conll_idx]
          buff[i][j] = (word, ) + words[word] + poss[pos] + verbs[verb] + is_verbs[is_verb] + srls[srl] + verb_senses[verb_sense]
          buff[i][j] += tuple([is_verbs_index[j]]) # is_Verbs_index is same as is_verbs in this case
          buff[i][j] += tuple([verb, srl, verb_sense])
          #print(buff[i][j])
    elif self.model_type == "SenseDisamb":
      words, poss, srls, verbs, is_verbs, verb_senses = self.vocabs
      for i, sent in enumerate(buff):
        is_verbs_index = [1 if item[is_verbs.conll_idx] == '1' else 0 for item in sent]
        for j, token in enumerate(sent):
          word, pos, verb, is_verb, srl, verb_sense = token[words.conll_idx], token[poss.conll_idx], token[verbs.conll_idx], token[is_verbs.conll_idx], token[srls.conll_idx], token[verb_senses.conll_idx]
          buff[i][j] = (word, ) + words[word] + poss[pos] + verbs[verb] + is_verbs[is_verb] + srls[srl] + verb_senses[verb_sense]
          buff[i][j] += tuple([is_verbs_index[j]]) # is_Verbs_index is same as is_verbs in this case
          buff[i][j] += tuple([verb, srl, verb_sense])
          #print(buff[i][j])
    elif self.model_type == "Parser":
      words, poss, rels = self.vocabs
      for i, sent in enumerate(buff):
        for j, token in enumerate(sent):
          # Reformat the input file : word_id word ppos head deprel
          word, pos, head, rel = token[words.conll_idx], token[poss.conll_idx], token[3], token[rels.conll_idx]
          buff[i][j] = (word,) + words[word] + poss[pos] + (int(head),) + rels[rel]
        sent.insert(0, ('root', Vocab.ROOT, Vocab.ROOT, 0 , Vocab.ROOT))
    elif self.model_type == "MultiTask":
      pass
    else:
      print("Unknown type In buff process")
    return buff

  #=============================================================
  def reset(self, sizes):
    """"""

    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return

  #=============================================================
  def rebucket(self):
    """"""

    buff = self._file_iterator.next()
    len_cntr = Counter()

    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)

    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return

  #=============================================================
  def _finalize(self):
    """"""

    self._metabucket._finalize()
    return

  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""

    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      if batch_size == 0:
        n_splits = 1
      #elif not self.minimize_pads:
      #  n_splits = max(len(bucket) // batch_size, 1)
      #  if bucket.size > 100:
      #    n_splits *= 2
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]
      feas = self[bkt_idx].fea[bkt_mb]
      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))
      feed_dict = {
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: data[:,:maxlen,target_idxs]
      }
      yield feed_dict, sents, feas

  #=============================================================
  def get_minibatches2(self, batch_size, input_idxs, target_idxs):
    """"""

    bkt_lens = np.empty(len(self._metabucket))
    for i, bucket in enumerate(self._metabucket):
      bkt_lens[i] = len(bucket)

    total_sents = np.sum(bkt_lens)
    bkt_probs = bkt_lens / total_sents
    n_sents = 0
    while n_sents < total_sents:
      n_sents += batch_size
      bkt = np.random.choice(self._metabucket._buckets, p=bkt_probs)
      data = bkt.data[np.random.randint(len(bkt), size=batch_size)]
      if bkt.size > 100:
        for data_ in np.array_split(data, 2):
          feed_dict = {
            self.inputs: data_[:,:,input_idxs],
            self.targets: data_[:,:,target_idxs]
          }
          yield feed_dict
      else:
        feed_dict = {
          self.inputs: data[:,:,input_idxs],
          self.targets: data[:,:,target_idxs]
        }
        yield feed_dict

  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts

  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
  def sentsNum(self):
    return sum([len(bucket) for bucket in self._metabucket])
