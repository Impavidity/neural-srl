#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.sensedisamb.base_sensedisamb import BaseSenseDisamb
from lib.linalg import linear


class SenseDisamb(BaseSenseDisamb):

  def __call__(self, dataset, moving_params=None):
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:, :, 0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1, 1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params

    word_inputs = vocabs[0].embedding_lookup(inputs[:, :, 0], inputs[:, :, 1], keep_prob=self.word_keep_prob, moving_params=self.moving_params)
    pos_inputs = vocabs[1].embedding_lookup(inputs[:, :, 2], moving_params=self.moving_params)
    verb_inputs = vocabs[3].embedding_lookup(inputs[:, :, 3], moving_params=self.moving_params)
    is_verb_inputs = vocabs[4].embedding_lookup(inputs[:, :, 4], moving_params=self.moving_params)

    top_recur = tf.concat(2, [word_inputs, pos_inputs, verb_inputs, is_verb_inputs])
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)

    predicate_token = tf.to_int32(tf.equal(inputs[:,:,4], vocabs[4]['1']))


    # Mask predicate hidden representation out
    predicate_h = tf.mul(top_recur, tf.to_float(tf.expand_dims(predicate_token, 2)))
    # Reduce dimension
    predicate_h = tf.reduce_sum(predicate_h, axis=1) # b x dim1
    # Mask the predicate Embedding out
    predicate_em = tf.mul(verb_inputs, tf.to_float(tf.expand_dims(predicate_token, 2)))
    # Reduce dimension
    predicate_em = tf.reduce_sum(predicate_em, axis=1)  # b x dim2
    hid = tf.concat(1, [predicate_h, predicate_em])


    # Mask the truth out
    truth_index = tf.mul(targets, predicate_token)
    # Reduce Dimension
    truth_index = tf.reduce_sum(truth_index, axis=1)

    with tf.variable_scope("MLP", reuse=reuse):
      hid = self.SenseMLP(hid, set_size=len(vocabs[5]))
      cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(hid, truth_index)
      predictions1D = tf.to_int32(tf.argmax(hid, 1))
      correct1D = tf.to_float(tf.equal(predictions1D, truth_index))
      n_correct = tf.reduce_sum(correct1D)
      loss = tf.reduce_sum(cross_entropy1D) / tf.reduce_sum(tf.to_float(tf.greater(self.sequence_lengths, -1)))

    output = {}
    output["loss"] = loss
    output["predictions"] = predictions1D
    output["n_correct"] = n_correct
    output["predicate_token"] = predicate_token

    return output




