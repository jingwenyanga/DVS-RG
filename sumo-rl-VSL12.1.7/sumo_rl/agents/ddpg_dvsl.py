#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 01:36:24 2018

@author: wuyuankai
"""

from __future__ import division
# from Priority_Replay import SumTree, Memory
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np
import time

import os
import sys

tools = '/usr/share/sumo/tools'
sys.path.append(tools)


EP_MAX = 600
LR_A = 0.0002  # learning rate for actor
LR_C = 0.0005  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 64
BATCH_SIZE = 32

RENDER = False


###############################  DDPG  ####################################


class VSL_DDPG_PR(object):
    def __init__(self, a_dim, s_dim, ):
        # self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)
        self.td = self.R + GAMMA * q_ - q

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        print("self.pointer",self.pointer)
        if self.pointer<BATCH_SIZE:
            pass
        else:
        #        tree_idx, bt, ISWeights = self.memory.sample(BATCH_SIZE)
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    #    def store_transition(self, s, a, r, s_):
    #        transition = np.hstack((s, a, r, s_))
    #        self.memory.store(transition)
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            neta = tf.layers.dense(s, 60, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(neta, self.a_dim, activation=tf.nn.sigmoid, name='l2', trainable=trainable,
                                use_bias=False)
            return tf.multiply(a, 8, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 50
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            netc = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(netc, 1, trainable=trainable)

    def savemodel(self, ):
        self.saver.save(self.sess, 'ddpg_networkss_withoutexplore/' + 'ddpg.ckpt')

    def loadmodel(self, ):
        loader = tf.train.import_meta_graph('ddpg_networkss_withoutexplore/ddpg.ckpt.meta')
        loader.restore(self.sess, tf.train.latest_checkpoint("ddpg_networkss_withoutexplore/"))
