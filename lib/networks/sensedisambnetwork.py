#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle as pkl
from srleval import evaluate
from srleval import SenseEvaluate
import json

import numpy as np
import tensorflow as tf

from remove import removeDul

from lib import models
from lib import optimizers

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset

"""
This is the Network for Predicate Sense Disambiguation
This is the first step in the Pipline System.
After training the network and get the result, we need to
 use the sense for final evaluation.
"""

class SenseDisambNetwork(Configurable):

  def __init__(self, ar, model, *args, **kwargs):

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')

    name = kwargs.pop('name', model.__name__)
    super(SenseDisambNetwork, self).__init__(*args, **kwargs)
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
                   (self.srl_file, 3, 'Srls'),
                   (self.verb_file, 4, 'Verbs'),
                   (self.is_verb_file, 5, 'IsVerbs'),
                   (self.verb_sense_file, 6, 'VerbSenses')]
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    load_embed_file=(not i),
                    global_step=self.global_step)
      self._vocabs.append(vocab)
    print("###################### Data #################")
    print("There are %d words in training set" % (len(self.words)-3))
    print("There are %d pos tag in training set" % (len(self.poss)-3))
    for pos in self.poss:
      print(pos, self.poss[pos])
    print("There are %d verbs in training set" % (len(self.verbs)-3))
    #for verb in self.verbs:
    #  print(verb, self.verbs[verb])
    print("There are %d presence tag in training set" % (len(self.is_verbs)-3))
    #for is_verb in self.is_verbs:
    #  print(is_verb, self.is_verbs[is_verb])
    print("There are %d srls in training set" % (len(self.srls)-3))
    for srl in self.srls:
      print(srl, self.srls[srl])
    print("There are %d senses in training set" % (len(self.verb_senses)-3))
    for sense in self.verb_senses:
      print(sense, self.verb_senses[sense])
    print("########################Data##################")


    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')
    self._oodset = Dataset(self.ood_file, self._vocabs, model, self._config, name="OODset")
    print("There are %d sentences in training set" % (self._trainset.sentsNum()))
    print("There are %d sentences in validation set" % (self._validset.sentsNum()))
    print("There are %d sentences in testing set" % (self._testset.sentsNum()))
    print("There are %d sentences in ood set" % (self._oodset.sentsNum()))

    self._ops = self._gen_ops()

    return
#============================================================================

  def _gen_ops(self):

    optimizer = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    train_output = self._model(self._trainset)

    l2_loss = self.l2_reg * tf.add_n([tf.nn.l2_loss(matrix) for matrix in tf.get_collection("Weights")])
    recur_loss = self.recur_reg * tf.add_n(tf.get_collection('recur_losses')) if self.recur_reg else self.model.ZERO
    covar_loss = self.covar_reg * tf.add_n(tf.get_collection('covar_losses')) if self.covar_reg else self.model.ZERO
    ortho_loss = self.ortho_reg * tf.add_n(tf.get_collection('ortho_losses')) if self.ortho_reg else self.model.ZERO
    regularization_loss = recur_loss + covar_loss + ortho_loss
    if self.recur_reg or self.covar_reg or self.ortho_reg or 'pretrain_loss' in train_output:
      optimizer2 = optimizers.RadamOptimizer(self._config)
      pretrain_loss = train_output.get('pretrain_loss', self.model.ZERO)
      pretrain_op = optimizer2.minimize(pretrain_loss + regularization_loss)
    else:
      pretrain_loss = self.model.ZERO
      pretrain_op = self.model.ZERO

    train_op = optimizer.minimize(train_output['loss'] + l2_loss + regularization_loss)
    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=optimizer)
    test_output = self._model(self._testset, moving_params=optimizer)
    ood_output = self._model(self._oodset, moving_params=optimizer)
    ops = {}
    ops['pretrain_op'] = [pretrain_op,
                          pretrain_loss,
                          recur_loss,
                          covar_loss,
                          ortho_loss]
    ops['train_op'] = [train_op,
                       train_output['loss'] + l2_loss + regularization_loss,
                       train_output['predictions'],
                       train_output['n_correct']
                       ]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['predictions'],
                       valid_output['n_correct']
                       ]
    ops['test_op'] = [test_output['loss'],
                      test_output['predictions'],
                      test_output['n_correct']
                    ]
    ops['ood_op'] = [ood_output['loss'],
                      ood_output['predictions'],
                      ood_output['n_correct']
                      ]
    ops['optimizer'] = optimizer

    return ops
 #==========================================================================
  def debug(self, sents, inputs, targets):
    for sent, feature, target in zip(sents, inputs, targets):
      for word, fea, tar in zip(sent, feature, target):
        print(word, self.words[(fea[0], fea[1])], self.poss[fea[2]], self.verbs[fea[3]], self.is_verbs[fea[4]],
              self.verb_senses[tar], end=" ")
        print()
      print()
      break

#=============================================================================

  def train(self, sess):

    saver = tf.train.Saver(name=self.name, max_to_keep=1)
    validate_every = self.validate_every
    print_every = self.print_every
    try:
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_iters = 0
      total_train_iters = sess.run(self.global_step)
      best_acc = 0
      test_acc = 0
      ood_acc = 0
      while True:
        for j, (feed_dict, _sent, _feas) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          #self.debug(_sent, train_inputs, train_targets)
          _, loss, predictions, n_correct= sess.run(self.ops['train_op'], feed_dict=feed_dict)
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_iters += 1
          total_train_iters += 1
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_sents
            print("Iter: %6d Loss: %5.2f Accuracy: %5.2f Sents: %d" %(total_train_iters, train_loss, train_accuracy, n_train_sents))
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_iters = 0
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            print("## Validation: %8d" % int(total_train_iters/validate_every))
            acc = self.test(sess, validate=True, ood=False)
            if (acc > best_acc):
              best_acc = acc
              print("## Update the model")
              print("## TEST")
              saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                           latest_filename=self.name.lower(), global_step=self.global_epoch)
              test_acc = self.test(sess, validate=False, ood=False)
              print("## OOD")
              ood_acc = self.test(sess, validate=False, ood=True)
            print("## Currently the accuracy %5.2f" % (best_acc))
            print("## The Test set accuracy %5.2f" % (test_acc))
            print("## The OOD set accuracy %5.2f" % (ood_acc))
        print("[epoch] ", sess.run(self._global_epoch.assign_add(1.)))
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    return
  #==========================================================================
  def test(self, sess, validate=False, ood=False):
    """"""
    filename = None
    minibatches = None
    dataset = None
    op = None
    if validate and not ood:
      filename = self.valid_file
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.ops['valid_op']
    elif not validate and not ood:
      filename = self.test_file
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.ops['test_op']
    elif not validate and ood:
      filename = self.ood_file
      minibatches = self.ood_minibatches
      dataset = self._oodset
      op = self.ops['ood_op']
    else:
      print("Not Supported Situation in Test")
      exit()


    all_predictions = [[]]
    all_sents = [[]]
    all_feature = [[]]
    all_targets = [[]]
    bkt_idx = 0
    for (feed_dict, sents, _feas) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      #self.debug(sents, mb_inputs, mb_targets)
      loss, predictions, n_correct = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(predictions)
      all_sents[-1].extend(sents)
      all_targets[-1].extend(mb_targets)
      all_feature[-1].extend(_feas)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        assert(len(all_predictions)==len(all_sents))
        assert(len(all_predictions)==len(all_targets))
        assert(len(all_predictions)==len(all_feature))
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
          all_feature.append([])
          all_targets.append([])
    acc = self.model_calc(dataset, all_sents, all_feature, all_targets, all_predictions, self._vocabs, validate, ood)
    return acc

  #=========================================================================
  def model_calc(self, dataset, sents, features, targets, predictions, vocabs, validate, ood):
    fout = None
    if validate and not ood:
      fout = open(self.save_dir+"/SenseDisambDevOut", "w")
    elif not validate and not ood:
      fout = open(self.save_dir+"/SenseDisambTestOut", "w")
    elif not validate and ood:
      fout = open(self.save_dir + "/SenseDisambOODOut", "w")
    else:
      print("Unsupported Mode in Result Calc")
      exit()
    for bkt_idx, idx in dataset._metabucket.data:
      datas = dataset._metabucket[bkt_idx].data[idx]
      words = sents[bkt_idx][idx]
      feas = features[bkt_idx][idx]
      truths = targets[bkt_idx][idx]
      predict = predictions[bkt_idx][idx]
      for i, (word, fea, truth, data) in enumerate(zip(words, feas, truths, datas)):
        if (fea[3] == "-1"):
          ptag = "-1"
        else:
          ptag = self.verb_senses[predict]
        if (fea[3] != "-1" and self.verbs[word] == Vocab.UNK) or (fea[3] != "-1" and predict <= Vocab.UNK):
          ptag = "01"
        tup = (
          str(i+1),
          word,
          fea[3],
          ptag
        )
        fout.write("%s\t%s\t%s\t%s\n" % tup)
      fout.write("\n")
    fout.flush()
    fout.close()
    acc = -1
    if validate and not ood:
      acc = SenseEvaluate(self.save_dir+"/SenseDisambDevOut", self.valid_file)
    elif not validate and not ood:
      acc = SenseEvaluate(self.save_dir + "/SenseDisambTestOut", self.test_file)
    elif not validate and ood:
      acc = SenseEvaluate(self.save_dir + "/SenseDisambOODOut", self.ood_file)

    print("## Sense Accuracy %5.2f" % (acc))
    return acc


  #=========================================================================
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

  # =============================================================
  def test_minibatches(self):
    """"""

    return self._testset.get_minibatches(self.test_batch_size,
                                         self.model.input_idxs,
                                         self.model.target_idxs,
                                         shuffle=False)
  #===============================================================
  def ood_minibatches(self):
    """"""

    return self._oodset.get_minibatches(self.test_batch_size,
                                         self.model.input_idxs,
                                         self.model.target_idxs,
                                         shuffle=False)
  # =============================================================

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
  def poss(self):
    return self._vocabs[1]
  @property
  def verbs(self):
    return self._vocabs[3]
  @property
  def is_verbs(self):
    return self._vocabs[4]
  @property
  def srls(self):
    return self._vocabs[2]
  @property
  def verb_senses(self):
    return self._vocabs[5]
  @property
  def ops(self):
    return self._ops