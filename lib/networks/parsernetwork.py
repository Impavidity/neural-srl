#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle as pkl
from parserEval import parserEval
import json

import numpy as np
import tensorflow as tf

from remove import removeDul

from lib import models
from lib import optimizers

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset

class ParserNetwork(Configurable):
  """"""

  #=============================================================
  def __init__(self, model, *args, **kwargs):
    """"""

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')

    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(ParserNetwork, self).__init__(*args, **kwargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    with open(os.path.join(self.save_dir, 'config.cfg'), 'w') as f:
      self._config.write(f)

    self._global_step = tf.Variable(0., trainable=False)
    self._global_epoch = tf.Variable(0., trainable=False)
    self._model = model(self._config, global_step=self.global_step)

    self._vocabs = []
    vocab_files = [(self.word_file, 1, 'Words'),
                   (self.pos_file, 2, 'Poss'),
                   (self.dep_file, 4, 'Deps')]
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    load_embed_file=(not i),
                    global_step=self.global_step)
      self._vocabs.append(vocab)

    print("Begin to load the Dataset")

    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')
    self._oodset = Dataset(self.ood_file, self._vocabs, model, self._config, name='OODset')
    print("There are %d sentences in training set" % (self._trainset.sentsNum()))
    print("There are %d sentences in validation set" % (self._validset.sentsNum()))
    print("There are %d sentences in testing set" % (self._testset.sentsNum()))
    print("There are %d sentences in ood set" % (self._oodset.sentsNum()))

    print("Loaded the Dataset")

    self._ops = self._gen_ops()

    return

  #=============================================================
  def train_minibatches(self):
    """"""

    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)

  #=============================================================
  def valid_minibatches(self):
    """"""

    return self._validset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)

  #=============================================================
  def test_minibatches(self):
    """"""

    return self._testset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)

  #=============================================================
  # assumes the sess has already been initialized
  def train(self, sess):
    """"""
    saver = tf.train.Saver(name=self.name, max_to_keep=1)
    train_iters = self.train_iters
    print_every = self.print_every
    validate_every = self.validate_every
    try:
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      best_uas = 0
      best_las = 0
      test_uas = 0
      test_las = 0
      total_train_iters = sess.run(self.global_step)
      while True:
        for j, (feed_dict, _, _feas) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          _, loss, n_correct, n_tokens = sess.run(self.ops['train_op'], feed_dict=feed_dict)
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_tokens += n_tokens
          n_train_iters += 1
          total_train_iters += 1
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            print("## Validation: %8d" % (int(total_train_iters / validate_every)))
            uas, las = self.test(sess, validate=True)

            print("## Validation uas: %f  las: %f" %(uas, las))
            if las > best_las:
              best_las = las
              best_uas = uas
              saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                         latest_filename=self.name.lower(), global_step=self.global_epoch)
              test_uas, test_las = self.test(sess, validate=False)
            print("## Currently the best validate UAS : %5.2f LAS : %5.2f" % (best_uas, best_las))
            print("## Responding Test UAS : %5.2f LAS : %5.2f " % (test_uas, test_las))
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_tokens
            print('Iter: %6d Train loss: %.4f  Train acc: %5.2f%%  Sents: %d' % (total_train_iters, train_loss, train_accuracy, n_train_sents))
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_tokens = 0
            n_train_iters = 0
        print("[epoch] ",sess.run(self._global_epoch.assign_add(1.)))
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    with open(os.path.join(self.save_dir, 'scores.txt'), 'w') as f:
      pass
    self.test(sess, validate=True)
    return

  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test(self, sess, validate=False):
    """"""

    if validate:
      filename = self.valid_file
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.ops['test_op'][0]
    else:
      filename = self.test_file
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.ops['test_op'][1]

    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    for (feed_dict, sents, _feas) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      mb_probs = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(self.model.validate(mb_inputs, mb_targets, mb_probs))
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      for bkt_idx, idx in dataset._metabucket.data:
        data = dataset._metabucket[bkt_idx].data[idx][1:]
        preds = all_predictions[bkt_idx][idx]
        words = all_sents[bkt_idx][idx]
        for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
          tup = (
            i+1,
            word,
            self.tags[pred[3]] if pred[3] != -1 else self.tags[datum[2]],
            self.tags[pred[4]] if pred[4] != -1 else self.tags[datum[3]],
            str(pred[5]) if pred[5] != -1 else str(datum[4]),
            self.rels[pred[6]] if pred[6] != -1 else self.rels[datum[5]],
            str(pred[7]) if pred[7] != -1 else '_',
            self.rels[pred[8]] if pred[8] != -1 else '_',
          )
          f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
        f.write('\n')
      f.flush()
      f.close()
    uas, las = parserEval(os.path.join(self.save_dir, os.path.basename(filename)), filename)
    # with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
    #   s, _ = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)), punct=self.model.PUNCT)
    #   f.write(s)
    # return s
    return uas, las

  #==============================================================
  def _gen_ops(self):
    """"""

    optimizer = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    train_output = self._model(self._trainset)

    l2_loss = self.l2_reg * tf.add_n([tf.nn.l2_loss(matrix) for matrix in tf.get_collection('Weights')]) if self.l2_reg else self.model.ZERO
    recur_loss = self.recur_reg * tf.add_n(tf.get_collection('recur_losses')) if self.recur_reg else self.model.ZERO
    covar_loss = self.covar_reg * tf.add_n(tf.get_collection('covar_losses')) if self.covar_reg else self.model.ZERO
    ortho_loss = self.ortho_reg * tf.add_n(tf.get_collection('ortho_losses')) if self.ortho_reg else self.model.ZERO
    regularization_loss = recur_loss + covar_loss + ortho_loss
    if self.recur_reg or self.covar_reg or self.ortho_reg or 'pretrain_loss' in train_output:
      optimizer2 = optimizers.RadamOptimizer(self._config)
      pretrain_loss = train_output.get('pretrain_loss', self.model.ZERO)
      pretrain_op = optimizer2.minimize(pretrain_loss+regularization_loss)
    else:
      pretrain_loss = self.model.ZERO
      pretrain_op = self.model.ZERO

    train_op = optimizer.minimize(train_output['loss']+l2_loss+regularization_loss)
    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=optimizer)
    test_output = self._model(self._testset, moving_params=optimizer)

    ops = {}
    ops['pretrain_op'] = [pretrain_op,
                          pretrain_loss,
                          recur_loss,
                          covar_loss,
                          ortho_loss]
    ops['train_op'] = [train_op,
                       train_output['loss']+l2_loss+regularization_loss,
                       train_output['n_correct'],
                       train_output['n_tokens']]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['n_correct'],
                       valid_output['n_tokens'],
                       valid_output['predictions']]
    ops['test_op'] = [valid_output['probabilities'],
                      test_output['probabilities']]
    ops['optimizer'] = optimizer

    return ops

  #=============================================================
  @property
  def global_step(self):
    return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def model(self):
    return self._model
  @property
  def words(self):
    return self._vocabs[0]
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def ops(self):
    return self._ops
