#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.multitask.base_multitask import BaseMultiTask
from lib import linalg

#***************************************************************
class MultiTask(BaseMultiTask):
  """"""

  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""

    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.tokens_to_keep3D_compute_loss = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    word_inputs = vocabs[0].embedding_lookup(inputs[:, :, 0], inputs[:, :, 1], keep_prob=self.word_keep_prob, moving_params=self.moving_params)
    pos_inputs = vocabs[1].embedding_lookup(inputs[:, :, 2], moving_params=self.moving_params)
    verb_inputs = vocabs[4].embedding_lookup(inputs[:, :, 3], moving_params=self.moving_params)
    is_verb_inputs = vocabs[5].embedding_lookup(inputs[:, :, 4], moving_params=self.moving_params)

    #top_recur = tf.concat(2, [word_inputs, pos_inputs, verb_inputs, is_verb_inputs])
    top_recur = tf.concat(2, [word_inputs, pos_inputs])
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _be = self.RNN(top_recur)

    be_parser = be_srler = _be
    # Modify Here
    word_pre = top_mlp = top_recur
    # top_mlp = top_recur
    # End of Modify


    # Add another layer to parser here
    for i in xrange(self.n_parser_recur):
      with tf.variable_scope("ParserRNN%d" % i, reuse=reuse):
        top_mlp, be_parser = self.RNN(top_mlp)

    word_com = tf.concat(2, [word_pre, verb_inputs, is_verb_inputs])

    for i in xrange(self.n_srl_recur):
      with tf.variable_scope("SRLCOMRNN%d" % i, reuse=reuse):
        word_com, be_srler = self.RNN(word_com)

    if self.n_mlp > 0:
      with tf.variable_scope('MLP0', reuse=reuse):
        dep_mlp, head_dep_mlp, rel_mlp, head_rel_mlp = self.MultiUpdateMLP(top_mlp, n_splits=4)
      for i in xrange(1,self.n_mlp):
        with tf.variable_scope('DepMLP%d' % i, reuse=reuse):
          dep_mlp = self.MLP(dep_mlp)
        with tf.variable_scope('HeadDepMLP%d' % i, reuse=reuse):
          head_dep_mlp = self.MLP(head_dep_mlp)
        with tf.variable_scope('RelMLP%d' % i, reuse=reuse):
          rel_mlp = self.MLP(rel_mlp)
        with tf.variable_scope('HeadRelMLP%d' % i, reuse=reuse):
          head_rel_mlp = self.MLP(head_rel_mlp)
    else:
      dep_mlp = head_dep_mlp = rel_mlp = head_rel_mlp = top_mlp

    if self.complicated_loss:
      with tf.variable_scope('loss_weight_para', reuse=reuse):
        loss_para_input = tf.concat(1, [be_parser, be_srler])
        loss_para = self.MLP4LossWeight(loss_para_input)
        loss_para4parser = tf.mul(self.tokens_to_keep3D_compute_loss, loss_para)
        loss_para4srler = self.tokens_to_keep3D - loss_para4parser



    with tf.variable_scope('Parses', reuse=reuse):
      parse_logits = self.bilinear_classifier(dep_mlp, head_dep_mlp, add_bias1=True)
      if self.complicated_loss:
        parse_output = self.complicated_output(parse_logits, targets[:,:,0], loss_para4parser)
      else:
        parse_output = self.output(parse_logits, targets[:,:,0])
      if moving_params is None:
        predictions = targets[:,:,0]
      else:
        predictions = parse_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      if self.complicated_loss:
        rel_output = self.complicated_output(rel_logits, targets[:,:,1], loss_para4parser)
      else:
        rel_output = self.output(rel_logits, targets[:,:,1])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)

    # Modifying
    # srl_top_recur = self.embed_concat(word_inputs, tag_inputs)
    # for i in xrange(self.n_recur):
    #   with tf.variable_scope('SRLRNN%d' % i, reuse=reuse):
    #    srl_top_recur, _ = self.RNN(srl_top_recur)
    # word_com = tf.concat(2, [srl_top_recur, verb_inputs, is_verb_inputs] + context)
    # End of Modifying


    predicate_token = tf.to_int32(tf.equal(inputs[:, :, 4], vocabs[5]['1']))
    # Mask predicate hidden representation out
    predicate_h = tf.mul(word_com, tf.to_float(tf.expand_dims(predicate_token, 2)))
    # print(predicate_h.get_shape().as_list())
    # Reduce dimension
    predicate_h = tf.reduce_sum(predicate_h, axis=1)
    # print(predicate_h.get_shape().as_list())
    # Broadcasting
    predicate_h = tf.mul(tf.expand_dims(predicate_h, 1), self.tokens_to_keep3D)
    # print(predicate_h.get_shape().as_list())
    # Concat
    classifier_input = tf.concat(2, [predicate_h, word_com])

    # Mask predicate embedding out
    predicate_em = tf.mul(verb_inputs, tf.to_float(tf.expand_dims(predicate_token, 2)))
    # Reduce dimension
    predicate_em = tf.reduce_sum(predicate_em, axis=1)  # b x dim
    # b x dim -> b x r x dim
    predicate_em = tf.pack([predicate_em] * len(vocabs[3]), axis=1)
    # Role Representation

    role_em_list = vocabs[3].embedding_lookup(range(len(vocabs[3])), moving_params=self.moving_params)
    batch_list = tf.to_float(tf.greater(self.sequence_lengths, -1))
    # print(batch_list.get_shape().as_list())
    # batch_list = tf.pack([batch_list]*len(vocabs[2]), axis=1)
    # print(batch_list.get_shape().as_list())
    role_em_list = tf.mul(tf.to_float(tf.expand_dims(batch_list, 2)), role_em_list)
    para_input = tf.concat(2, [predicate_em, role_em_list])

    with tf.variable_scope('SRLClassifier_para', reuse=reuse):
      para = self.MLP4SRLWeight(para_input)
      result_dist = tf.batch_matmul(classifier_input, para, adj_y=True)
      if self.complicated_loss:
        srl_output = self.complicated_output(result_dist, targets[:,:,2], loss_para4srler)
      else:
        srl_output = self.output(result_dist, targets[:,:,2])




    output = {}
    output['probabilities'] = tf.tuple([parse_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions_dep'] = parse_output['predictions']
    output["predictions_srl"] = srl_output['predictions']
    output['correct'] = parse_output['correct'] * rel_output['correct']
    output['tokens'] = parse_output['tokens']
    output['n_correct_dep'] = tf.reduce_sum(output['correct'])
    output["n_correct_srl"] = srl_output["n_correct"]
    output['n_tokens'] = self.n_tokens
    output['accuracy_dep'] = output['n_correct_dep'] / output['n_tokens']
    output['accuracy_srl'] = output['n_correct_srl'] / output['n_tokens']
    output['loss_parser'] = parse_output['loss'] + rel_output['loss']
    output['loss_srl'] = srl_output["loss"]
    output['sequence_lengths'] = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])

    return output

  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""

    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
