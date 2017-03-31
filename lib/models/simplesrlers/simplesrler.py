#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.simplesrlers.base_simplesrler import BaseSimpleSrlers
from lib.linalg import linear


class SimpleSrler(BaseSimpleSrlers):

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
        #self.embed_concat()
        # top_recur = tf.concat(2, [word_inputs, pos_inputs, verb_inputs, is_verb_inputs]+ context)
        ###################### Modify Here
        # top_recur = tf.concat(2, [word_inputs, pos_inputs, verb_inputs, is_verb_inputs])
        top_recur = tf.concat(2, [word_inputs, pos_inputs])
        ###################### End of Modify
        for i in xrange(self.n_recur):
          with tf.variable_scope('RNN%d' % i, reuse=reuse):
            top_recur, _ = self.RNN(top_recur)

        ##################### Modify Here
        top_recur = tf.concat(2, [top_recur, verb_inputs, is_verb_inputs])
        for i in xrange(self.n_srl_recur):
          with tf.variable_scope('SRLRNN%d' % i, reuse=reuse):
            top_recur, _ = self.RNN(top_recur)
            ##################### End of Modify

        predicate_token = tf.to_int32(tf.equal(inputs[:, :, 4], vocabs[4]['1']))

        # Mask predicate hidden representation out
        predicate_h = tf.mul(top_recur, tf.to_float(tf.expand_dims(predicate_token, 2)))
        # print(predicate_h.get_shape().as_list())
        # Reduce dimension
        predicate_h = tf.reduce_sum(predicate_h, axis=1)
        # print(predicate_h.get_shape().as_list())
        # Broadcasting
        predicate_h = tf.mul(tf.expand_dims(predicate_h, 1), self.tokens_to_keep3D)
        # print(predicate_h.get_shape().as_list())
        # Concat
        classifier_input = tf.concat(2, [predicate_h, top_recur])

        # Mask predicate embedding out
        predicate_em = tf.mul(verb_inputs, tf.to_float(tf.expand_dims(predicate_token, 2)))
        # Reduce dimension
        predicate_em = tf.reduce_sum(predicate_em, axis=1)  # b x dim
        # b x dim -> b x r x dim
        predicate_em = tf.pack([predicate_em] * len(vocabs[2]), axis=1)
        # Role Representation

        role_em_list = vocabs[2].embedding_lookup(range(len(vocabs[2])), moving_params=self.moving_params)
        batch_list = tf.to_float(tf.greater(self.sequence_lengths, -1))
        # print(batch_list.get_shape().as_list())
        # batch_list = tf.pack([batch_list]*len(vocabs[2]), axis=1)
        # print(batch_list.get_shape().as_list())
        role_em_list = tf.mul(tf.to_float(tf.expand_dims(batch_list, 2)), role_em_list)
        para_input = tf.concat(2, [predicate_em, role_em_list])

        with tf.variable_scope('SRLClassifier_para', reuse=reuse):
          para = self.MLP4SRLWeight(para_input)
          result_dist = tf.batch_matmul(classifier_input, para, adj_y=True)
          srl_output = self.output(result_dist, targets)




        '''
        with tf.variable_scope('MLP0', reuse=reuse):
          argument = self.MLP(top_mlp)
        verb_inputs = tf.expand_dims(verb_inputs, 1)
        context = []
        for i in range(self.windonw_length):
          context.append(tf.expand_dims(vocabs[0].embedding_lookup(inputs[:, -1, 5 + i * 2], inputs[:, -1, 5 + i * 2 + 1],
                                                              moving_params=self.moving_params),1))
        com = tf.concat(2, [verb_inputs]+context)

        with tf.variable_scope('MLP1', reuse=reuse):
          predicate = self.MLP(com)
        with tf.variable_scope('GraphSRL', reuse=reuse):
          srl_logits = self.bilinear_classifier_srl(argument, predicate, len(vocabs[2]), add_bias1=True)
          srl_output = self.output(srl_logits, targets)
        '''



        #decode_output = self.CRFDecode(crf_output, targets[:,:,0])

        output = {}
        output["loss"] = srl_output["loss"]
        output["n_tokens"] = self.n_tokens
        output["predictions"] = srl_output['predictions']
        output["n_correct"] = srl_output["n_correct"]
        output["sequence_lengths"] = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
        # output["prob"] = result_dist
        return output

        # decode_output = self.CRFDecode(crf_output, targets[:,:,0])
        # output = {}
        # output["loss"] = crf_output["loss"]
        # output["n_tokens"] = self.n_tokens
        # output["crf_output"] = crf_output
        # output["sequence_lengths"] = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])

        # return output




