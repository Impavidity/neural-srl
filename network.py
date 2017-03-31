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

# TODO make the optimizer class inherit from Configurable
# TODO bayesian hyperparameter optimization
# TODO start a UD tagger/parser pipeline
#***************************************************************

class SimpleSRLNetwork(Configurable):

  def __init__(self, model, *args, **kwargs):

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')

    name = kwargs.pop('name', model.__name__)
    super(SimpleSRLNetwork, self).__init__(*args, **kwargs)
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
                   (self.verb_sense_file, 6, 'VerbSense')]
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
    print("There are %d senses in training set") % (len(self.verb_senses) - 3)
    for sense in self.verb_senses:
      print(sense, self.verb_senses[sense])
    print("########################Data##################")

    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')
    self._oodset = Dataset(self.ood_file, self._vocabs, model, self._config, name="OODset")

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
    ops = {}
    ops['pretrain_op'] = [pretrain_op,
                          pretrain_loss,
                          recur_loss,
                          covar_loss,
                          ortho_loss]

    ops['train_op'] = [train_op,
                       train_output['loss'] + l2_loss + regularization_loss,
                       train_output['predictions'],
                       train_output['n_tokens'],
                       train_output['n_correct'],
                       train_output['sequence_lengths']
                       #train_output['prob']
                       ]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['predictions'],
                       valid_output['n_tokens'],
                       valid_output['n_correct'],
                       valid_output['sequence_lengths']
                       ]
    ops['test_op'] = [test_output['loss'],
                      test_output['predictions'],
                      test_output['n_tokens'],
                      test_output['n_correct'],
                      test_output['sequence_lengths']
                    ]


    ops['optimizer'] = optimizer

    return ops
 #==========================================================================
  def debug(self, sents, inputs, targets):
    for sent, feature, target in zip(sents, inputs, targets):
      for word, fea, tar in zip(sent, feature, target):
        print(word, self.words[(fea[0], fea[1])], self.poss[fea[2]], self.verbs[fea[3]], self.is_verbs[fea[4]],
              self.srls[tar], end=" ")
        for item in range(int(len(fea[5:]) / 2)):
          print(fea[item * 2 + 5], fea[item * 2 + 5 + 1], self.words[(fea[item * 2 + 5], fea[item * 2 + 5 + 1])],
                end=" ")
        print()
      print()

#=============================================================================

  def train(self, sess):

    saver = tf.train.Saver(name=self.name, max_to_keep=1)
    train_iters = self.train_iters
    validate_every = self.validate_every
    print_every = self.print_every
    try:
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      total_train_iters = sess.run(self.global_step)
      best_f = 0
      best_p = 0
      best_r = 0
      test_p = 0
      test_r = 0
      test_f = 0
      while True:
        for j, (feed_dict, _sent, _feas) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          #self.debug(_sent, train_inputs, train_targets)
          _, loss, predictions, n_tokens, n_correct, sequence_lengths, prob= sess.run(self.ops['train_op'], feed_dict=feed_dict)
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_tokens += n_tokens
          n_train_iters += 1
          total_train_iters += 1
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_tokens
            print("Iter: %6d Loss: %5.2f Accuracy: %5.2f Sents: %d" %(total_train_iters, train_loss, train_accuracy, n_train_sents))
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_tokens = 0
            n_train_iters = 0
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            print("## Validation: %8d" % int(total_train_iters/validate_every))
            p, r, f= self.test(sess, validate=True)
            if (f>best_f):
              best_f = f
              best_p = p
              best_r = r
              print("## Update the model")
              print("## TEST")
              saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                           latest_filename=self.name.lower(), global_step=self.global_epoch)
              test_p, test_r, test_f = self.test(sess, validate=False)
            print("## Currently the best F %5.2f P %5.2f R %5.2f" %(best_f, best_p, best_r))
            print("## The Test set F %5.2f P %5.2f R %5.2f" % (test_f, test_p, test_r))
        print("[epoch] ", sess.run(self._global_epoch.assign_add(1.)))
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    self.test(sess, validate=True)
    return
  #==========================================================================
  def test(self, sess, validate=False):
    """"""

    if validate:
      filename = self.valid_file
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.ops['valid_op']
    else:
      filename = self.test_file
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.ops['test_op']

    all_predictions = []
    all_sents = []
    all_feature = []
    all_targets = []
    all_sequence_lengths = []
    for (feed_dict, sents, _feas) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      loss, predictions, n_tokens, n_correct, sequence_lengths = sess.run(op, feed_dict=feed_dict)
      all_predictions.append(predictions)
      all_sents.append(sents)
      all_targets.append(mb_targets)
      all_feature.append(_feas)
      all_sequence_lengths.append(sequence_lengths)
    p, r, f = self.model_calc(all_sequence_lengths, all_sents, all_feature, all_targets, all_predictions, self._vocabs, validate)
    return p, r, f

  #=========================================================================
  def model_calc(self, sequence_lengths, sents, features, targets, predictions, vocabs, validate):
    fout = None
    if validate:
      fout1 = open(self.save_dir+"/DevOut", "w")
      fout2 = open(self.save_dir + "/DevGold", "w")
    else:
      fout1 = open(self.save_dir+"/TestOut", "w")
      fout2 = open(self.save_dir + "/TestGold", "w")
    for batch_lengths, batch_sents, batch_feature, batch_targets, batch_predictions in zip(sequence_lengths, sents, features, targets, predictions):
      for length, words, feas, truths, predicts in zip(batch_lengths, batch_sents, batch_feature, batch_targets, batch_predictions):
        count = 0
        for word_id, word, fea, truth, predict in zip(range(length), words, feas, truths, predicts):
          #print(fea)
          count += 1
          if fea[0] == '1' or fea[0] == 1:
            if self.srls[predict] == "_":
              fout1.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t_\n" % (count, word, fea[1]+".01"))
            elif "+" in self.srls[predict]:
              sense = self.srls[predict].split("+")[0]
              tag = self.srls[predict].split("+")[1]
              fout1.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t%s\n" % (count, word, fea[1] + "."+sense, tag))
            elif self.srls[predict][0]=='0':
              fout1.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t_\n" % (count, word, fea[1]+"."+self.srls[predict]))
            else:
              fout1.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t%s\n" % (count, word, fea[1] + ".01" , self.srls[predict]))
            if not "+" in fea[2]:
              fout2.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t_\n" % (count, word, fea[1] + "." + fea[2]))
            else:
              sense = fea[2].split("+")[0]
              tag = fea[2].split("+")[1]
              fout2.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t%s\n" % (count, word, fea[1] + "." + sense, tag))
          elif fea[0] == '0' or fea[0] == 0:
            fout1.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t%s\n" % (count, word, self.srls[predict]))
            fout2.write("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t%s\n" % (count, word, fea[2]))
          else:
            print("Error")
            exit()
        fout1.write('\n')
        fout2.write('\n')
    fout1.flush()
    fout1.close()
    fout2.flush()
    fout2.close()
    if validate:
      p, r, f = evaluate(self.save_dir+"/DevOut", self.save_dir+"/DevGold")
    else:
      p, r, f = evaluate(self.save_dir + "/TestOut", self.save_dir+"/TestGold")

    print("## F1: %f P: %f R: %f" % (f, p, r))
    return p, r, f


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
#==============================================================

"""
This is the Network for Predicate Sense Disambiguation
This is the first step in the Pipline System.
After training the network and get the result, we need to
 use the sense for final evaluation.
"""

class SenseDisambNetwork(Configurable):

  def __init__(self, model, *args, **kwargs):

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
    self.test(sess, validate=True)
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

#***************************************************************
if __name__ == '__main__':
  """"""

  import argparse

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--validate', action='store_true')
  argparser.add_argument('--ood', action='store_true')
  argparser.add_argument('--load', action='store_true')
  # store_true : if it is declared, then set it as true
  args, extra_args = argparser.parse_known_args()
  #args: Namespace(load=False, model='Parser', pretrain=False, test=False)
  #extra_args: ["--some","xxxx"]
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}
  #Attention: You need to specify the model type in the config file and the command line at the same time and match
  if 'model_type' not in cargs:
    print("You need to specify the model_type")
    exit()
  print('*** '+cargs['model_type']+' ***')

  model = getattr(models, cargs['model_type'])

  if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not (args.test or args.load or args.validate):
    raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  if (args.test or args.load or args.validate) and 'save_dir' in cargs:
    cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')


  network = None


  if cargs['model_type'] == "SimpleSrler":
    cargs.pop("model_type","")
    network = SimpleSRLNetwork(model, **cargs)
  elif cargs['model_type'] == "SenseDisamb":
    cargs.pop("model_type","")
    network = SenseDisambNetwork(model, **cargs)

  else:
    print("Unsupported Model")
    exit()


  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    if not args.test and not args.validate:
      if args.load:
        '''
        os.system('echo Training: > %s/HEAD' % network.save_dir)
        saver = tf.train.Saver(name=network.name)
        saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
        network.is_load = True
        if os.path.isfile(os.path.join(network.save_dir, 'history.pkl')):
          with open(os.path.join(network.save_dir, 'history.pkl')) as f:
            network.history = pkl.load(f)
        '''
        pass # We do not support restart training mechanism currently
      else:
        os.system('echo Loading: >> %s/HEAD' % network.save_dir)

      network.train(sess)
    else:
      pass

      os.system('echo Testing: >> %s/HEAD' % network.save_dir)
      saver = tf.train.Saver(name=network.name)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      if network.model_type == "SenseDisamb":
        if args.test and not args.validate and not args.ood:
          acc = network.test(sess, validate=False, ood=False)
          print("## TESTING ##")
          print("Sense Accuracy %5.2f" % (acc))
        elif not args.test and not args.ood and args.validate:
          acc = network.test(sess, validate=True, ood=False)
          print("## VALIDATION ##")
          print("Sense Accuracy %5.2f" % (acc))
        elif not args.test and not args.validate and args.ood:
          acc = network.test(sess, validate=True, ood=False)
          print("## OOD ##")
          print("Sense Accuracy %5.2f" % (acc))
