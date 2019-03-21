"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import python_getdents 
from scipy import spatial
from sklearn.decomposition import PCA
from itertools import islice
import itertools


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, summary, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    print('######## length of update_gradient_vars: %d\n' % len(update_gradient_vars))
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='Adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='Adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='Adam':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSProp':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='Momentum':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        elif optimizer=='SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)

        else:
            raise ValueError('Invalid optimization algorithm')

        gvs = opt.compute_gradients(total_loss, update_gradient_vars)

    ### gradient clip for handling the gradient exploding
    gradslist, varslist = zip(*gvs)
    grads_clip, _ = tf.clip_by_global_norm(gradslist, 5.0)
    #grads_clip = [(tf.clip_by_value(grad, -1.0, 1.0),var) for grad, var in grads]

    # Apply gradients.
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #apply_gradient_op = opt.apply_gradients(grads_clip, global_step=global_step)
    apply_gradient_op = opt.apply_gradients(zip(grads_clip, varslist), global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        #for var in tf.trainable_variables():
        for var in update_gradient_vars:
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    variables_averages_op = variable_averages.apply(update_gradient_vars)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #with tf.control_dependencies([apply_gradient_op]):
         train_op = tf.no_op(name='train')
    
    print('######## length of update_gradient_vars: %d\n' % len(update_gradient_vars))
    return train_op, gvs, grads_clip

def train_layerwise(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    print('######## length of update_gradient_vars: %d\n' % len(update_gradient_vars))

    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    #     if optimizer=='ADAGRAD':
    #         opt = tf.train.AdagradOptimizer(learning_rate)
    #     elif optimizer=='ADADELTA':
    #         opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    #     elif optimizer=='ADAM':
    #         opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    #     elif optimizer=='RMSPROP':
    #         opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    #     elif optimizer=='MOM':
    #         opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    #     else:
    #         raise ValueError('Invalid optimization algorithm')
    # grads = opt.compute_gradients(total_loss, update_gradient_vars)
    # Apply gradients.
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    beta = 100
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt1 = tf.train.AdagradOptimizer(learning_rate/beta)
            opt2 = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt1 = tf.train.AdadeltaOptimizer(learning_rate/beta, rho=0.9, epsilon=1e-6)
            opt2 = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt1 = tf.train.AdamOptimizer(learning_rate/beta, beta1=0.9, beta2=0.999, epsilon=0.1)
            opt2 = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt1 = tf.train.RMSPropOptimizer(learning_rate/beta, decay=0.9, momentum=0.9, epsilon=1.0)
            opt2 = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt1 = tf.train.MomentumOptimizer(learning_rate/beta, 0.9, use_nesterov=True)
            opt2 = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        ## glabal learning rate


        ## layer-wise learning rate for updating the gradients
        update_gradient_vars1 = []
        update_gradient_vars2 = []
        for var in update_gradient_vars:
            if not ('Embeddings/' in var.op.name or 'Centralisation/' in var.op.name or 'centers' in var.op.name or 'centers_cts' in var.op.name):
                update_gradient_vars1.append(var)
            else:
                update_gradient_vars2.append(var)
        grads1 = opt1.compute_gradients(total_loss, update_gradient_vars1)
        grads2 = opt2.compute_gradients(total_loss, update_gradient_vars2)
        #grads = tf.group(grads1, grads2)
        grads = grads1 + grads2

        apply_gradient_op1 = opt1.apply_gradients(grads1, global_step=global_step)
        apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
        apply_gradient_op = tf.group(apply_gradient_op1, apply_gradient_op2)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

  












