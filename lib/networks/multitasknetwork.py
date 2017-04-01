#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from srleval import evaluate
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

class MultiTaskNetwork(Configurable):
  """"""

  # =============================================================
  def __init__(self, model, *args, **kwargs):
    """"""

    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')

    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(MultiTaskNetwork, self).__init__(*args, **kwargs)
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
                   (self.dep_file, 4, 'Deps'),
                   (self.srl_file, 5, 'Srls'),
                   (self.verb_file, 6, 'Verbs'),
                   (self.is_verb_file, 7, "IsVerbs")]
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    load_embed_file=(not i),
                    global_step=self.global_step)
      self._vocabs.append(vocab)

    print("###################### Data #################")
    print("There are %d words in training set" % (len(self.words) - 3))
    #for word in self.words:
    #  print(word, self.words[word])
    print("There are %d pos tag in training set" % (len(self.poss) - 3))
    for pos in self.poss:
      print(pos, self.poss[pos])
    print("There are %d verbs in training set" % (len(self.verbs) - 3))
    #for verb in self.verbs:
    #  print(verb, self.verbs[verb])
    print("There are %d presence tag in training set" % (len(self.is_verbs) - 3))
    for is_verb in self.is_verbs:
      print(is_verb, self.is_verbs[is_verb])
    print("There are %d srls in training set" % (len(self.srls) - 3))
    for srl in self.srls:
      print(srl, self.srls[srl])
    print("There are %d dep in training set" % (len(self.deps) - 3))
    for dep in self.deps:
      print(dep, self.deps[dep])
    print("########################Data##################")

    print("Begin to load the Dataset")

    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')



    print("Loaded the Dataset")

    self._ops = self._gen_ops()

    return

  # =============================================================
  def train_minibatches(self):
    """"""

    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)

  # =============================================================
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

  # =============================================================================
  def debug(self, sents, inputs, targets):
    for sent, feature, target in zip(sents, inputs, targets):
      for word, fea, tar in zip(sent, feature, target):
        print(word, self.words[(fea[0], fea[1])], self.poss[fea[2]], self.verbs[fea[3]], self.is_verbs[fea[4]],
              tar[0], self.deps[tar[1]], self.srls[tar[2]], end=" ")
        print()
      print()
  # =============================================================================
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
      n_train_correct_dep = 0
      n_train_correct_srl = 0
      n_train_tokens = 0
      n_train_iters = 0
      total_train_iters = sess.run(self.global_step)
      best_score = 0
      best_p = 0
      best_r = 0
      best_f = 0
      best_las = 0
      best_uas = 0
      test_p = 0
      test_r = 0
      test_f = 0
      test_las = 0
      test_uas = 0
      test_Macro = 0

      if self.is_load:
        with open(self.save_dir+"/best_history") as json_data:
          best_score, best_p, best_r, best_f, best_las, best_uas, test_Macro, test_p, test_r, test_f, test_las, test_uas = json.load(json_data)
      while True:
        for j, (feed_dict, _sent, _feas) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          # Check Data Input
          #self.debug(_sent, train_inputs, train_targets)
          if self.stacking == False and self.complicated_loss == False:
            _, loss, n_correct_dep, n_correct_srl, predictions_dep, predictions_srl, \
              n_tokens = sess.run(self.ops['train_op'], feed_dict=feed_dict)
          if self.stacking == True and self.complicated_loss == False:
            if sess.run(self._global_epoch) < 120:
              _, loss, n_correct_dep, n_correct_srl, predictions_dep, predictions_srl, \
                n_tokens = sess.run(self.ops['train_op_stacking1'], feed_dict=feed_dict)
            else:
              _, loss, n_correct_dep, n_correct_srl, predictions_dep, predictions_srl, \
                n_tokens = sess.run(self.ops['train_op_stacking2'], feed_dict=feed_dict)
          if self.stacking == False and self.complicated_loss == True:
            _, loss, n_correct_dep, n_correct_srl, predictions_dep, predictions_srl, \
              n_tokens = sess.run(self.ops['train_op_complicated_loss'], feed_dict=feed_dict)
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct_dep += n_correct_dep
          n_train_correct_srl += n_correct_srl
          n_train_tokens += n_tokens
          n_train_iters += 1
          total_train_iters += 1

          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy_dep = 100 * n_train_correct_dep / n_train_tokens
            train_accuracy_srl = 100 * n_train_correct_srl / n_train_tokens
            print(
              '%6d) Train loss: %.4f    Train dep acc: %5.2f%%  Train srl acc: %5.2f%%  '
              % (total_train_iters, train_loss, train_accuracy_dep, train_accuracy_srl))
            train_loss = 0
            n_train_sents = 0
            n_train_correct_dep = 0
            n_train_correct_srl = 0
            n_train_tokens = 0
            n_train_iters = 0

          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            print("## Validation: %8d" % int(total_train_iters / validate_every))
            uas, las, p, r, f = self.test(sess, validate=True)
            print("## Validation UAS: %5.2f LAS: %5.2f P: %5.2f R: %5.2f F: %5.2f" % (uas, las, p, r, f))
            LMP = 0.5 * p + 0.5 * las
            LMR = 0.5 * r + 0.5 * las
            Macro = 2 * LMP * LMR / (LMP + LMR)
            print("## Validation Macro %5.2f" % (Macro))
            if Macro > best_score:
              best_score = Macro
              best_p = p
              best_r = r
              best_f = f
              best_las = las
              best_uas = uas
              print("## Update Model")
              saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                         latest_filename=self.name.lower(), global_step=self.global_epoch)
              print("## Test")
              test_uas, test_las, test_p, test_r, test_f = self.test(sess, validate=False)
              test_LMP = 0.5 * test_p + 0.5 * test_las
              test_LMR = 0.5 * test_r + 0.5 * test_las
              test_Macro = 2 * test_LMP * test_LMR / (test_LMP + test_LMR)
              fupdate = open(self.save_dir+'/best_history','w')
              fupdate.write(json.dumps([best_score, best_p, best_r, best_f, best_las, best_uas, test_Macro, test_p, test_r, test_f, test_las, test_uas]))
            print("## Currently the best F %5.2f P %5.2f R %5.2f LAS %5.2f UAS %5.2f Macro %5.2f" % (best_f, best_p, best_r, best_las, best_uas, best_score))
            print("## The Test set F %5.2f P %5.2f R %5.2f LAS %5.2f UAS %5.2f Macro %5.2f" % (test_f, test_p, test_r, test_las, test_uas, test_Macro))

        print("[epoch ]",sess.run(self._global_epoch.assign_add(1.)))
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

  # =============================================================
  def model_calc(self, sents, features, targets, predictions, vocabs, validate):
    fout = None
    if validate:
      fout1 = open(self.save_dir + "/DevOut", "w")
      fout2 = open(self.save_dir + "/DevGold", "w")
    else:
      fout1 = open(self.save_dir + "/TestOut", "w")
      fout2 = open(self.save_dir + "/TestGold", "w")
    sents_num = 0
    for batch_sents, batch_feature, batch_predictions in zip(sents, features, predictions):
      for words, feas,  predicts in zip(batch_sents, batch_feature, batch_predictions):
        count = 0
        sents_num += 1
        for word, fea, predict in zip(words, feas[1:], predicts[1:]):
          # print(fea)
          count += 1
          if fea[1] == '<pad>':
            break
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
              fout1.write(
                "%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\tY\t%s\t%s\n" % (count, word, fea[1] + ".01" , self.srls[predict]))
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
        if fea[1]!='<pad>':
          fout1.write('\n')
          fout2.write('\n')
    fout1.flush()
    fout1.close()
    fout2.flush()
    fout2.close()
    if validate:
      p, r, f = evaluate(self.save_dir + "/DevOut", self.save_dir + "/DevGold")
    else:
      p, r, f = evaluate(self.save_dir + "/TestOut", self.save_dir + "/TestGold")
    print("*** There are %d Sentences ***" % (sents_num))

    return p, r, f

  # =============================================================
  # TODO make this work if lines_per_buff isn't set to 0
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

    all_predictions = [[]]
    all_sents = [[]]
    all_srl_feas = [[]]
    all_srl_pres = [[]]
    bkt_idx = 0

    for (feed_dict, sents, _feas) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      n_correct_dep, n_correct_srl, mb_probs, predictions_srl, n_tokens = sess.run(op, feed_dict=feed_dict)

      sequence = predictions_srl
      all_srl_feas[-1].extend(_feas)
      all_srl_pres[-1].extend(sequence)
      all_predictions[-1].extend(self.model.validate(mb_inputs, mb_targets, mb_probs, sequence))
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
          all_srl_feas.append([])
          all_srl_pres.append([])

    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      for bkt_idx, idx in dataset._metabucket.data:
        data = dataset._metabucket[bkt_idx].data[idx][1:]
        preds = all_predictions[bkt_idx][idx]
        words = all_sents[bkt_idx][idx]
        for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
          tup = (
            i + 1,
            word,
            self.poss[datum[2]],
            self.poss[datum[2]],
            str(pred[4]) if pred[4] != -1 else '_',
            self.deps[pred[5]] if pred[5] != -1 else '_',
            str(pred[6]) if pred[6] != -1 else '_',
            self.deps[pred[7]] if pred[7] != -1 else '_',
          )
          if self.deps[pred[7]] == "<UNK>":
            break
          f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
        if self.deps[pred[7]] != "<UNK>":
          f.write('\n')

      f.flush()
      f.close()
    removeDul(os.path.join(self.save_dir, os.path.basename(filename)))
    if validate:
      uas, las = parserEval(os.path.join(self.save_dir, os.path.basename(filename)+"_rm"), "../Conll09/dep.conll.dev")
    else:
      if self.ood:
        uas, las = parserEval(os.path.join(self.save_dir, os.path.basename(filename) + "_rm"), "../Conll09/dep.conll.ood")
      else:
        uas, las = parserEval(os.path.join(self.save_dir, os.path.basename(filename) + "_rm"), "../Conll09/dep.conll.test")
    # with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
    #   s, _ = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)+"_rm"), punct=self.model.PUNCT)
    #  f.write(s)
    # return uas, las, p, r, f
    # las = float(s.strip().split()[3])
    # uas = float(s.strip().split()[1])
    #print("## Validation: uas %5.2f las %5.2f" %(uas,las))
    p, r, f = self.model_calc(all_sents, all_srl_feas, None, all_srl_pres, None, validate=validate)
    return uas,las,p,r,f

  # =============================================================
  def _gen_ops(self):
    """"""

    optimizer = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    optimizer_stacking1 = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    optimizer_stacking2 = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    optimizer_complicated_loss = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    train_output = self._model(self._trainset)

    l2_loss = self.l2_reg * tf.add_n(
      [tf.nn.l2_loss(matrix) for matrix in tf.get_collection('Weights')]) if self.l2_reg else self.model.ZERO
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

    train_op = optimizer.minimize(self.weighted_parser * train_output['loss_parser'] + train_output['loss_srl'] + l2_loss + regularization_loss)
    train_op_stacking1 = optimizer.minimize(train_output['loss_parser'] + l2_loss + regularization_loss)
    train_op_stacking2 = optimizer.minimize(train_output['loss_srl'] + l2_loss + regularization_loss)
    train_op_complicated_loss = optimizer.minimize(train_output['loss_parser'] + train_output['loss_srl'] + l2_loss + regularization_loss)
    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=optimizer)
    test_output = self._model(self._testset, moving_params=optimizer)

    ops = {}
    ops['pretrain_op'] = [pretrain_op,
                          pretrain_loss,
                          recur_loss,
                          covar_loss,
                          ortho_loss]
    #_, loss, n_correct_dep, n_correct_srl, predictions_dep, predictions_srl, n_tokens
    ops['train_op'] = [train_op,
                       train_output['loss_parser'] + train_output['loss_srl'] + l2_loss + regularization_loss,
                       train_output['n_correct_dep'],
                       train_output['n_correct_srl'],
                       train_output['predictions_dep'],
                       train_output['predictions_srl'],
                       train_output['n_tokens']
                       ]
    ops['train_op_stacking1'] = [train_op_stacking1,
                       train_output['loss_parser'] + l2_loss + regularization_loss,
                       train_output['n_correct_dep'],
                       train_output['n_correct_srl'],
                       train_output['predictions_dep'],
                       train_output['predictions_srl'],
                       train_output['n_tokens']]
    ops['train_op_stacking2'] = [train_op_stacking2,
                       train_output['loss_srl'] + l2_loss + regularization_loss,
                       train_output['n_correct_dep'],
                       train_output['n_correct_srl'],
                       train_output['predictions_dep'],
                       train_output['predictions_srl'],
                       train_output['n_tokens']]
    ops['train_op_complicated_loss'] = [train_op_complicated_loss,
                       train_output['loss_parser'] + train_output['loss_srl'] + l2_loss + regularization_loss,
                       train_output['n_correct_dep'],
                       train_output['n_correct_srl'],
                       train_output['predictions_dep'],
                       train_output['predictions_srl'],
                       train_output['n_tokens']]
    ops['valid_op'] = [valid_output['n_correct_dep'],
                       valid_output['n_correct_srl'],
                       valid_output['probabilities'],
                       valid_output['predictions_srl'],
                       valid_output['n_tokens']]
    ops['test_op'] = [ test_output['n_correct_dep'],
                       test_output['n_correct_srl'],
                       test_output['probabilities'],
                       test_output['predictions_srl'],
                       test_output['n_tokens']]


    ops['optimizer'] = optimizer

    return ops

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
  def deps(self):
    return self._vocabs[2]

  @property
  def srls(self):
    return self._vocabs[3]

  @property
  def verbs(self):
    return self._vocabs[4]

  @property
  def is_verbs(self):
    return self._vocabs[5]

  @property
  def ops(self):
    return self._ops