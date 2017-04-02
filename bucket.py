#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from configurable import Configurable

#***************************************************************
class Bucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(Bucket, self).__init__(*args, **kwargs)
    self._size = None
    self._data = None
    self._sents = None
    self._fea = None
    return
  
  #=============================================================
  def reset(self, size):
    """"""
    
    self._size = size
    self._data = []
    self._sents = []
    self._fea = []
    return
  
  #=============================================================
  def add(self, sent):
    """"""
    
    if isinstance(self._data, np.ndarray):
      raise TypeError("The buckets have already been finalized, you can't add more")
    if len(sent) > self.size and self.size != -1:
      raise ValueError('Bucket of size %d received sequence of len %d' % (self.size, len(sent)))
    words = None
    idxs = None
    idxf = None

    if self.model_type == "SimpleSrler" or self.model_type == "SenseDisamb":
      words = [word[0] for word in sent]
    elif self.model_type == "Parser" or self.model_type == "MultiTask":
      words = [word[0] for word in sent][1:] # Remove root
    else:
      print("Unsupported Mode in Add Feature")
      exit()

    if self.model_type == "SimpleSrler" or self.model_type == "SenseDisamb" or self.model_type == "MultiTask":
      idxs = [word[1:-4] for word in sent]
    elif self.model_type == "Parser":
      idxs = [word[1:] for word in sent]
    else:
      print("Unsupported Mode in Add Feature")
      exit()

    if self.model_type == "SimpleSrler" or self.model_type == "SenseDisamb" or self.model_type == "MultiTask":
      idxf = [word[-4:] for word in sent]
    elif self.model_type == "Parser":
      idxf = None
    else:
      print("Unsupported Mode in Add Feature")
      exit()
    self._sents.append(words)
    self._data.append(idxs)
    self._fea.append(idxf)
    return len(self._data)-1
  
  #=============================================================
  def _finalize(self):
    """"""
    
    if self._data is None:
      raise ValueError('You need to reset the Buckets before finalizing them')
    
    if len(self._data) > 0:
      shape = (len(self._data), self.size, len(self._data[-1][-1]))
      data = np.zeros(shape, dtype=np.int32)
      for i, datum in enumerate(self._data):
        datum = np.array(datum)
        data[i, 0:len(datum)] = datum
      self._data = data
      self._sents = np.array(self._sents)
      self._fea = np.array(self._fea)
    else:
      self._data = np.zeros((0,1), dtype=np.float32)
      self._sents = np.zeros((0,1), dtype=str)
      self._fea = np.zeros((0,1), dtype=str)
    print('Bucket %s is %d x %d' % ((self._name,) + self._data.shape[0:2]))
    return
  
  #=============================================================
  def __len__(self):
    return len(self._data)
  
  #=============================================================
  @property
  def size(self):
    return self._size
  @property
  def data(self):
    return self._data
  @property
  def sents(self):
    return self._sents
  @property
  def fea(self):
    return self._fea