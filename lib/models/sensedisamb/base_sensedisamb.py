#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN

class BaseSenseDisamb(NN):

    def __call__(self, dataset, moving_paras=None):
        raise NotImplementedError

    def parse_argmax(self):
        raise NotImplementedError


    # =============================================================
    @property
    def input_idxs(self):
        return (0, 1, 2, 3, 4)


    @property
    def target_idxs(self):
        return (6)