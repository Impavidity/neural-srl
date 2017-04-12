#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf

from ConfigParser import SafeConfigParser

#***************************************************************
class Configurable(object):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""

    self._name = kwargs.pop('name', type(self).__name__)
    if args and kwargs:
      raise TypeError('Configurables must take either a config parser or keyword args')
    if args:
      if len(args) > 1:
        raise TypeError('Configurables take at most one argument')
    self.is_load = False
    self.complicated_loss = False
    self.stacking_dep = False
    self.stacking_srl = False
    self.srl_major = False
    self.dep_major = False
    self.stacking = False
    self.change = False
    if args:
      self._config = args[0]
    else:
      self._config = self._configure(**kwargs)
    return
  
  #=============================================================
  def _configure(self, **kwargs):
    """"""
    
    config = SafeConfigParser()
    config_files = [os.path.join('config', 'defaults.cfg'),
                    os.path.join('config', self.name.lower() + '.cfg'),
                    kwargs.pop('config_file', '')]
    config.read(config_files)
    for option, value in kwargs.iteritems():
      assigned = False
      for section in config.sections():
        if option in config.options(section):
          config.set(section, option, str(value))
          assigned = True
          break
      if not assigned:
        raise ValueError('%s is not a valid option.' % option)
    
    return config
  
  #=============================================================
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--config_file')
  argparser.add_argument('--data_dir')
  argparser.add_argument('--embed_dir')
  
  @property
  def name(self):
    return self._name
  argparser.add_argument('--name')
  
  #=============================================================
  # [OS]
  @property
  def model_type(self):
    return self._config.get('OS', 'model_type')
  argparser.add_argument('--model_type')
  @property
  def word_file(self):
    return self._config.get('OS', 'word_file')
  argparser.add_argument('--word_file')
  @property
  def pos_file(self):
    return self._config.get('OS', 'pos_file')
  argparser.add_argument('--pos_file')
  @property
  def verb_file(self):
    return self._config.get('OS', 'verb_file')
  argparser.add_argument('--verb_file')
  @property
  def is_verb_file(self):
    return self._config.get('OS', 'is_verb_file')
  argparser.add_argument('--is_verb_file')
  @property
  def verb_sense_file(self):
    return self._config.get('OS', 'verb_sense_file')
  argparser.add_argument('--verb_sense_file')
  @property
  def dep_file(self):
    return self._config.get('OS', 'dep_file')
  argparser.add_argument('--dep_file')
  @property
  def srl_file(self):
    return self._config.get('OS', 'srl_file')
  argparser.add_argument('--srl_file')
  @property
  def embed_file(self):
    return self._config.get('OS', 'embed_file')
  argparser.add_argument('--embed_file')
  @property
  def train_file(self):
    return self._config.get('OS', 'train_file')
  argparser.add_argument('--train_file')
  @property
  def valid_file(self):
    return self._config.get('OS', 'valid_file')
  argparser.add_argument('--valid_file')
  @property
  def test_file(self):
    return self._config.get('OS', 'test_file')
  argparser.add_argument('--test_file')
  @property
  def ood_file(self):
    return self._config.get('OS', 'ood_file')
  argparser.add_argument('--ood_file')
  @property
  def save_dir(self):
    return self._config.get('OS', 'save_dir')
  argparser.add_argument('--save_dir')
  @property
  def restore_from(self):
    return self._config.get('OS', 'restore_from')
  argparser.add_argument('--restore_from')
  @property
  def restore_name(self):
    return self._config.get('OS', 'restore_name')
  argparser.add_argument('--restore_name')
  @property
  def source_train(self):
    return self._config.get('OS', 'source_train')
  argparser.add_argument('--source_train')
  @property
  def source_dev(self):
    return self._config.get('OS', 'source_dev')
  argparser.add_argument('--source_dev')
  @property
  def source_test(self):
    return self._config.get('OS', 'source_test')
  argparser.add_argument('--source_test')
  @property
  def source_ood(self):
    return self._config.get('OS', 'source_ood')
  argparser.add_argument('--source_ood')

  
  #=============================================================
  # [Dataset]
  @property
  def cased(self):
    return self._config.getboolean('Dataset', 'cased')
  argparser.add_argument('--cased')
  @property
  def min_occur_count(self):
    return self._config.getint('Dataset', 'min_occur_count')
  argparser.add_argument('--min_occur_count')
  @property
  def minimize_pads(self):
    return self._config.getboolean('Dataset', 'minimize_pads')
  argparser.add_argument('--minimize_pads')
  @property
  def n_bkts(self):
    return self._config.getint('Dataset', 'n_bkts')
  argparser.add_argument('--n_bkts')
  @property
  def n_valid_bkts(self):
    return self._config.getint('Dataset', 'n_valid_bkts')
  argparser.add_argument('--n_valid_bkts')
  @property
  def lines_per_buffer(self):
    return self._config.getint('Dataset', 'lines_per_buffer')
  argparser.add_argument('--lines_per_buffer')
  @property
  def windonw_length(self):
    return self._config.getint('Dataset', 'window_length')
  argparser.add_argument('--window_length')
  
  #=============================================================
  # [Layers]
  @property
  def n_srl_recur(self):
    return self._config.getint('Layers', 'n_srl_recur')
  argparser.add_argument('--n_srl_recur')
  @property
  def n_recur(self):
    return self._config.getint('Layers', 'n_recur')
  argparser.add_argument('--n_recur')
  @property
  def n_parser_recur(self):
    return self._config.getint('Layers', 'n_parser_recur')
  argparser.add_argument('--n_parser_recur')
  @property
  def n_mlp(self):
    return self._config.getint('Layers', 'n_mlp')
  argparser.add_argument('--n_mlp')
  @property
  def recur_cell(self):
    from lib import rnn_cells
    return getattr(rnn_cells, self._config.get('Layers', 'recur_cell'))
  argparser.add_argument('--recur_cell')
  @property
  def recur_bidir(self):
    return self._config.getboolean('Layers', 'recur_bidir')
  argparser.add_argument('--recur_bidir')
  @property
  def forget_bias(self):
    if self._config.get('Layers', 'forget_bias') == 'None':
      from lib.linalg import sig_const
      return sig_const
    else:
      return self._config.getfloat('Layers', 'forget_bias')
  argparser.add_argument('--forget_bias')
  @property
  def context_from_lstm(self):
    return self._config.getboolean('Layers', 'context_from_lstm')
  argparser.add_argument('--context_from_lstm')
  
  #=============================================================
  # [Sizes]
  @property
  def embed_size(self):
    return self._config.getint('Sizes', 'embed_size')
  argparser.add_argument('--embed_size')
  @property
  def normal_size(self):
    return self._config.getint('Sizes', 'normal_size')
  argparser.add_argument('--normal_size')
  @property
  def combine_size(self):
    return self._config.getint('Sizes', 'combine_size')
  argparser.add_argument('--combine_size')
  @property
  def recur_size(self):
    return self._config.getint('Sizes', 'recur_size')
  argparser.add_argument('--recur_size')
  @property
  def mlp_size(self):
    return self._config.getint('Sizes', 'mlp_size')
  argparser.add_argument('--mlp_size')
  @property
  def combine_mlp_size(self):
    return self._config.getint('Sizes', 'combine_mlp_size')
  argparser.add_argument('--combine_mlp_size')
  @property
  def use_combine(self):
    return self._config.getboolean('Sizes', 'use_combine')
  argparser.add_argument('--use_combine')
  
  #=============================================================
  # [Functions]
  @property
  def recur_func(self):
    func = self._config.get('Functions', 'recur_func')
    if func == 'identity':
      return tf.identity
    else:
      return getattr(tf.nn, func)
  argparser.add_argument('--recur_func')
  @property
  def mlp_func(self):
    func = self._config.get('Functions', 'mlp_func')
    if func == 'identity':
      return tf.identity
    else:
      return getattr(tf.nn, func)
  argparser.add_argument('--mlp_func')
  
  #=============================================================
  # [Regularization]
  @property
  def l2_reg(self):
    return self._config.getfloat('Regularization', 'l2_reg')
  argparser.add_argument('--l2_reg')
  @property
  def recur_reg(self):
    return self._config.getfloat('Regularization', 'recur_reg')
  argparser.add_argument('--recur_reg')
  @property
  def covar_reg(self):
    return self._config.getfloat('Regularization', 'covar_reg')
  argparser.add_argument('--covar_reg')
  @property
  def ortho_reg(self):
    return self._config.getfloat('Regularization', 'ortho_reg')
  argparser.add_argument('--ortho_reg')
  
  #=============================================================
  # [Dropout]
  @property
  def drop_gradually(self):
    return self._config.getboolean('Dropout', 'drop_gradually')
  argparser.add_argument('--drop_gradually')
  @property
  def word_keep_prob(self):
    return self._config.getfloat('Dropout', 'word_keep_prob')
  argparser.add_argument('--word_keep_prob')
  @property
  def tag_keep_prob(self):
    return self._config.getfloat('Dropout', 'tag_keep_prob')
  argparser.add_argument('--tag_keep_prob')
  @property
  def rel_keep_prob(self):
    return self._config.getfloat('Dropout', 'rel_keep_prob')
  argparser.add_argument('--rel_keep_prob')
  @property
  def recur_keep_prob(self):
    return self._config.getfloat('Dropout', 'recur_keep_prob')
  argparser.add_argument('--recur_keep_prob')
  @property
  def ff_keep_prob(self):
    return self._config.getfloat('Dropout', 'ff_keep_prob')
  argparser.add_argument('--ff_keep_prob')
  @property
  def mlp_keep_prob(self):
    return self._config.getfloat('Dropout', 'mlp_keep_prob')
  argparser.add_argument('--mlp_keep_prob')
  
  #=============================================================
  # [Learning rate]
  @property
  def learning_rate(self):
    return self._config.getfloat('Learning rate', 'learning_rate')
  argparser.add_argument('--learning_rate')
  @property
  def decay(self):
    return self._config.getfloat('Learning rate', 'decay')
  argparser.add_argument('--decay')
  @property
  def decay_steps(self):
    return self._config.getfloat('Learning rate', 'decay_steps')
  argparser.add_argument('--decay_steps')
  @property
  def clip(self):
    return self._config.getfloat('Learning rate', 'clip')
  argparser.add_argument('--clip')
  @property
  def weighted_parser(self):
    return self._config.getfloat('Learning rate', 'weighted_parser')
  argparser.add_argument('--weighted_parser')
  @property
  def weighted_srler(self):
    return self._config.getfloat('Learning rate', 'weighted_srler')
  argparser.add_argument('--weighted_srler')

  
  #=============================================================
  # [Radam]
  @property
  def mu(self):
    return self._config.getfloat('Radam', 'mu')
  argparser.add_argument('--mu')
  @property
  def nu(self):
    return self._config.getfloat('Radam', 'nu')
  argparser.add_argument('--nu')
  @property
  def gamma(self):
    return self._config.getfloat('Radam', 'gamma')
  argparser.add_argument('--gamma')
  @property
  def epsilon(self):
    return self._config.getfloat('Radam', 'epsilon')
  argparser.add_argument('--epsilon')
  @property
  def chi(self):
    return self._config.getfloat('Radam', 'chi')
  argparser.add_argument('--chi')
  
  #=============================================================
  # [Training]
  @property
  def pretrain_iters(self):
    return self._config.getint('Training', 'pretrain_iters')
  argparser.add_argument('--pretrain_iters')
  @property
  def train_iters(self):
    return self._config.getint('Training', 'train_iters')
  argparser.add_argument('--train_iters')
  @property
  def train_batch_size(self):
    return self._config.getint('Training', 'train_batch_size')
  argparser.add_argument('--train_batch_size')
  @property
  def test_batch_size(self):
    return self._config.getint('Training', 'test_batch_size')
  argparser.add_argument('--test_batch_size')
  @property
  def validate_every(self):
    return self._config.getint('Training', 'validate_every')
  argparser.add_argument('--validate_every')
  @property
  def print_every(self):
    return self._config.getint('Training', 'print_every')
  argparser.add_argument('--print_every')
  @property
  def save_every(self):
    return self._config.getint('Training', 'save_every')
  argparser.add_argument('--save_every')
  @property
  def per_process_gpu_memory_fraction(self):
    return self._config.getfloat('Training', 'per_process_gpu_memory_fraction')
  argparser.add_argument('--per_process_gpu_memory_fraction')
  
