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
#import python_getdents
from scipy import spatial
from sklearn.decomposition import PCA
from itertools import islice
import itertools

#import h5py


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.sub(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss
  
def decov_loss(xs):
    """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'
    """
    x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
    m = tf.reduce_mean(x, 0, True)
    z = tf.expand_dims(x-m, 2)
    corr = tf.reduce_mean(tf.batch_matmul(z, tf.transpose(z, perm=[0,2,1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
    loss = 0.5*(corr_frob_sqr - corr_diag_sqr)
    return loss 
  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       This is not exactly the algorthim proposed in the paper, since the update/shift of the centers is not moving towards the
       centers (i.e. sum(Xi)/Nj, Xi is the element of class j) of the classes but the sum of the elements (sum(Xi)) in the class
    """
     #nrof_features = features.get_shape()[1]
     #centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
     #    initializer=tf.constant_initializer(0), trainable=False)
     #label = tf.reshape(label, [-1])
     #centers_batch = tf.gather(centers, label)
     #diff = (1 - alfa) * (centers_batch - features)
     #diff = alfa * (centers_batch - features)
     #centers = tf.scatter_sub(centers, label, diff)
    # loss = tf.nn.l2_loss(features - centers_batch)
    # return loss, centers, diff, centers_batch

    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       -- mzh 15/02/2017
       -- Correcting the center updating, center updates/shifts towards to the center of the correponding class with a weight:
       -- centers = centers- (1-alpha)(centers-sum(Xi)/Nj), where Xi is the elements of the class j, Nj is the number of the elements of class Nj
       -- code has been tested by the test script '../test/center_loss_test.py'
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
    loss = tf.nn.l2_loss(features - centers_batch)
    diff = (1 - alfa) * (centers_batch - features)

    ## update the centers
    label_unique, idx = tf.unique(label)
    zeros = tf.zeros_like(label_unique, tf.float32)
    ## calculation the repeat time of same label
    nrof_elements_per_class_clean = tf.scatter_update(centers_cts, label_unique, zeros)
    ones = tf.ones_like(label, tf.float32)
    ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class_update = tf.scatter_add(nrof_elements_per_class_clean, label, ones)
    ## nrof_elements_per_class_list is the number of the elements in each class in the batch
    nrof_elements_per_class_batch = tf.gather(nrof_elements_per_class_update, label)
    nrof_elements_per_class_batch_reshape = tf.reshape(nrof_elements_per_class_batch, [-1, 1])## reshape the matrix as 1 coloum no matter the dimension of the row (-1)
    diff_mean = tf.div(diff, nrof_elements_per_class_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)

    #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, nrof_elements_per_class_batch_reshape
    return loss, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape


def center_loss_similarity(features, label, alfa, nrof_classes):
    ## center_loss on cosine distance =1 - similarity instead of the L2 norm, i.e. Euclidian distance

    ## normalisation as the embedding vectors in order to similarity distance
    features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
    #loss = tf.nn.l2_loss(features - centers_batch) ## 0.5*(L2 norm)**2, L2 norm is the Euclidian distance
    similarity_all = tf.matmul(features, tf.transpose(tf.nn.l2_normalize(centers_batch, 1, 1e-10))) ## dot prodoct, cosine distance, similarity of x and y
    similarity_self = tf.diag_part(similarity_all)
    loss_x = tf.subtract(1.0, similarity_self)
    loss = tf.reduce_sum(loss_x) ## sum the cosine distance of each vector/tensor
    diff = (1 - alfa) * (centers_batch - features)
    ones = tf.ones_like(label, tf.float32)
    centers_cts = tf.scatter_add(centers_cts, label, ones) # counting the number of each class, the class is in the order of the [0,1,2,3,....] as initialzation
    centers_cts_batch = tf.gather(centers_cts, label)
    #centers_cts_batch_ext = tf.tile(centers_cts_batch, nrof_features)
    #centers_cts_batch_reshape = tf.reshape(centers_cts_batch_ext,[-1, nrof_features])
    centers_cts_batch_reshape = tf.reshape(centers_cts_batch, [-1,1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(centers_cts, label, zeros)
    #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, centers_cts_batch_reshape
    #return loss, centers, loss_x, similarity_all, similarity_self
    return loss, centers




def center_inter_loss_tf(features, nrof_features, label, alfa, nrof_classes): # tensorflow version
    """ center_inter_loss = center_loss/||Xi - centers(0,1,2,...i-1,i+1,i+2,...)||
        --mzh 22022017
    """
    # dim_features = features.get_shape()[1]
    # nrof_features = features.get_shape()[0]
    dim_features = features.get_shape()[1].value
    #nrof_features = features.get_shape()[0].value
    # dim_features = features.shape[1]
    # nrof_features = features.shape[0]
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    ## center_loss calculation
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers,label)  # get the corresponding center of each element in features, the list of the centers is in the same order as the features
    dist_centers = features - centers_batch
    dist_centers_sum = tf.reduce_sum(dist_centers**2,1)/2
    loss_center = tf.nn.l2_loss(dist_centers)

    ## calculation the repeat time of same label
    ones = tf.ones_like(label, tf.float32)
    centers_cts = tf.scatter_add(centers_cts, label, ones)  # counting the number of each class, the class is in the order of the [0,1,2,3,....] as initialzation
    centers_cts_batch = tf.gather(centers_cts, label)


    ## inter_center_loss calculation
    #label_unique, label_idx = tf.unique(label)
    #centers_batch1 = tf.gather(centers,label_unique)
    #nrof_classes_batch = centers_batch.get_shape()[0].value
    #centers_1D = tf.reshape(centers_batch1, [1, nrof_classes_batch * dim_features])
    centers_batch1 = tf.gather(centers,label)
    centers_1D = tf.reshape(centers_batch1, [1, nrof_features * dim_features])
    centers_2D = tf.tile(centers_1D, [nrof_features, 1])
    centers_3D = tf.reshape(centers_2D,[nrof_features, nrof_features, dim_features])
    features_3D = tf.reshape(features, [nrof_features, 1, dim_features])
    dist_inter_centers = features_3D - centers_3D
    dist_inter_centers_sum_dim = tf.reduce_sum(dist_inter_centers**2,2)/2
    centers_cts_batch_1D = tf.tile(centers_cts_batch,[nrof_features])
    centers_cts_batch_2D = tf.reshape(centers_cts_batch_1D, [nrof_features, nrof_features])
    dist_inter_centers_sum_unique = tf.div(dist_inter_centers_sum_dim, centers_cts_batch_2D)
    dist_inter_centers_sum_all = tf.reduce_sum(dist_inter_centers_sum_unique, 1)
    dist_inter_centers_sum = dist_inter_centers_sum_all - dist_centers_sum
    loss_inter_centers = tf.reduce_mean(dist_inter_centers_sum)

    ## total loss
    loss = tf.div(loss_center, loss_inter_centers)

    ## update centers
    diff = (1 - alfa) * (centers_batch - features)
#    ones = tf.ones_like(label, tf.float32)
#   centers_cts = tf.scatter_add(centers_cts, label, ones)  # counting the number of each class
#    centers_cts_batch = tf.gather(centers_cts, label)
    centers_cts_batch_reshape = tf.reshape(centers_cts_batch, [-1, 1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(centers_cts, label, zeros)
    # return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, centers_cts_batch_reshape
    return loss, centers,  loss_center, loss_inter_centers, center_cts_clear
    #return loss, centers, loss_center, loss_inter_centers, dist_inter_centers_sum_dim, centers_cts_batch_2D, dist_inter_centers_sum_unique, dist_inter_centers_sum_all, dist_inter_centers_sum, dist_inter_centers_sum, center_cts_clear

def center_inter_triplet_loss_tf(features, nrof_features, label, alfa, nrof_classes, beta): # tensorflow version
    """ center_inter_loss = center_loss/||Xi - centers(0,1,2,...i-1,i+1,i+2,...)||
        --mzh 22022017
    """
    dim_features = features.get_shape()[1].value
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    nrof_elements_per_class_list = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    ## center_loss calculation
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers,label)  # get the corresponding center of each element in features, the list of the centers is in the same order as the features
    dist_centers = features - centers_batch
    dist_centers_sum = tf.reduce_sum(dist_centers**2,1)/2
    loss_center = tf.nn.l2_loss(dist_centers)

    ## calculation the repeat time of same label
    ones = tf.ones_like(label, tf.float32)
    nrof_elements_per_class_list = tf.scatter_add(nrof_elements_per_class_list, label, ones)  # counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class = tf.gather(nrof_elements_per_class_list, label) #nrof_elements_per_class is the number of the elements in each class


    ## inter_center_loss calculation
    centers_batch1 = tf.gather(centers,label)
    centers_1D = tf.reshape(centers_batch1, [1, nrof_features * dim_features])
    centers_2D = tf.tile(centers_1D, [nrof_features, 1])
    centers_3D = tf.reshape(centers_2D,[nrof_features, nrof_features, dim_features])
    features_3D = tf.reshape(features, [nrof_features, 1, dim_features])
    dist_inter_centers = features_3D - centers_3D
    dist_inter_centers_sum_dim = tf.reduce_sum(dist_inter_centers**2,2)/2
    centers_cts_batch_1D = tf.tile(nrof_elements_per_class,[nrof_features])
    centers_cts_batch_2D = tf.reshape(centers_cts_batch_1D, [nrof_features, nrof_features])
    dist_inter_centers_sum_unique = tf.div(dist_inter_centers_sum_dim, centers_cts_batch_2D)
    dist_inter_centers_sum_all = tf.reduce_sum(dist_inter_centers_sum_unique, 1)
    dist_inter_centers_sum = dist_inter_centers_sum_all - dist_centers_sum
    loss_inter_centers = tf.reduce_mean(dist_inter_centers_sum)

    ## total loss
    loss = loss_center + (loss_center + beta*nrof_features - loss_inter_centers)

    ## update centers
    diff = (1 - alfa) * (centers_batch - features)
    centers_cts_batch_reshape = tf.reshape(nrof_elements_per_class, [-1, 1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(nrof_elements_per_class_list, label, zeros)
    return loss, centers,  loss_center, loss_inter_centers, center_cts_clear

def class_level_triplet_loss_tf(features, nrof_samples, label, alfa, nrof_classes, beta, gamma): # tensorflow version
    """ Class_level_Triple_loss, triple loss implemented on the centers of the class intead of the individual sample
        --mzh 30072017s
    """
    dim_features = features.get_shape()[1].value
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    nrof_elements_per_class = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    ## normalisation as the embedding vectors in order to similarity distance
    #features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    ## calculate centers
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    diff_within = centers_batch - features
    dist_within = tf.reduce_sum(diff_within**2/2, axis=1, keep_dims=True)
    dist_within_center = tf.reduce_sum(dist_within, axis=0) ## sum all the elements in the dist_centers_sum, dist_within_center is a scale

    ## inter_center_loss calculation
    label_unique,idx = tf.unique(label)
    centers_batch_unique = tf.gather(centers,label_unique)#select the centers corresponding to the batch samples, otherwise the whole centers will cause the overflow of the centers_2D
    nrof_centers_batch_unique = tf.shape(centers_batch_unique)[0]##very important, tf.shape() can be used to get the run-time dynamic tensor shape; however .get_shape() can only be used to get the shape of the static shape of the tensor
    centers_1D = tf.reshape(centers_batch_unique, [1, nrof_centers_batch_unique * dim_features])
    centers_2D = tf.tile(centers_1D, [nrof_samples, 1])
    centers_3D = tf.reshape(centers_2D, [nrof_samples,nrof_centers_batch_unique, dim_features])
    features_3D = tf.reshape(features, [nrof_samples, 1, dim_features])
    dist_inter_centers = features_3D - centers_3D
    dist_inter_centers_sum_dim = tf.reduce_sum(dist_inter_centers**2,2)/2 # calculate the L2 of the features, [nrof_samples, nrof_classes, feature_dimension]
    dist_inter_centers_sum_all =  tf.reduce_sum(dist_inter_centers_sum_dim)#sum all the elements in the dist_inter_centers_sum_dim

    ## total loss
    dist_within_2D = tf.tile(dist_within, [1, nrof_centers_batch_unique])
    dist_matrix = dist_within_2D + beta*tf.ones([nrof_samples, nrof_centers_batch_unique]) - gamma*dist_inter_centers_sum_dim
    loss_matrix = tf.maximum(dist_matrix, tf.zeros([nrof_samples, nrof_centers_batch_unique], tf.float32))
    loss_pre = tf.reduce_sum(loss_matrix) - nrof_samples*beta
    #loss = tf.divide(loss_pre, nrof_samples)
    loss = tf.divide(loss_pre, tf.multiply(tf.cast(nrof_samples, tf.float32),
                                           tf.cast(nrof_centers_batch_unique, tf.float32) - tf.cast(1, tf.float32)))

    #centers = tf.scatter_sub(centers, label, diff)

    ##update centers
    zeros = tf.zeros_like(label_unique, tf.float32)
    ## calculation the repeat time of same label
    nrof_elements_per_class_clean = tf.scatter_update(nrof_elements_per_class, label_unique, zeros)
    ones = tf.ones_like(label, tf.float32)
    ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class_update = tf.scatter_add(nrof_elements_per_class_clean, label, ones)
    ## nrof_elements_per_class_list is the number of the elements in each class in the batch
    nrof_elements_per_class_batch = tf.gather(nrof_elements_per_class_update, label)
    centers_cts_batch_reshape = tf.reshape(nrof_elements_per_class_batch, [-1, 1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)

    #return loss
    return loss, centers, dist_within_center, dist_inter_centers_sum_all, nrof_centers_batch_unique
    #return loss, loss_matrix, dist_matrix, dist_within_2D, dist_inter_centers_sum_dim, centers, dist_inter_centers, features_3D, centers_3D, centers_1D
    #return loss, dist_within_center, dist_inter_centers_sum_all, nrof_centers_batch


def class_level_triplet_loss_similarity_tf(features, nrof_samples, label, nrof_classes, beta): # tensorflow version
    """ Class_level_Triple_loss_similarity, triple loss implemented on the centers of the class intead of the individual sample, however here the distance cosine (representing the similarity) replaces the L2 distance inclass_level_triplet_loss_tf.
        --mzh 25062017
    """
    dim_features = features.get_shape()[1].value
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    nrof_elements_per_class = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    ## normalisation as the embedding vectors in order to similarity distance
    features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    #nrof_samples = label.get_shape()[0]
    #nrof_samples = tf.shape(label)[0]
    ## calculation the repeat time of same label
    ones = tf.ones_like(label, tf.float32)
    ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class = tf.scatter_add(nrof_elements_per_class, label, ones)
    ## nrof_elements_per_class_list is the number of the elements in each class
    nrof_elements_per_class_list = tf.gather(nrof_elements_per_class, label)

    ## calculate centers
    class_sum = tf.scatter_add(centers, label, features)
    centers = tf.divide(class_sum, nrof_elements_per_class[:,None])
    ##very important, tf.shape() can be used to get the run-time dynamic tensor shape; however .get_shape() can only be used to get the shape of the static shape of the tensor
    ## inter_center_loss calculation
    label_unique, idx = tf.unique(label)
    nrof_centers_batch = tf.shape(label_unique)[0]  ##very important, tf.shape() can be used to get the run-time dynamic tensor shape; however .get_shape() can only be used to get the shape of the static shape of the tensor
    centers_batch = tf.gather(centers, label_unique)


    ## class distance loss
    label = tf.reshape(label, [-1])
    #centers_list = tf.gather(centers,label)  # get the corresponding center of each element in features, the list of the centers is in the same order as the features
    ## dot prodoct, cosine distance, similarity of x and y
    similarity_all = tf.matmul(features, tf.transpose(centers_batch))

    centers_list = tf.gather(centers, label)
    similarity_all_nn = tf.matmul(features, tf.transpose(centers_list))
    similarity_self = tf.diag_part(similarity_all_nn)

    # n = tf.cast(tf.shape(label)[0],dtype=tf.int64)
    # a = tf.range(n, dtype=tf.int64)
    # #a = tf.ones_like(label)
    # #self_index_1 = tf.transpose(tf.stack([a[0:], label]))
    # self_index = tf.transpose(tf.stack([a, label]))
    # similarity_self = tf.gather_nd(similarity_all, self_index)

    similarity_self_mn = tf.tile(tf.transpose([similarity_self]), [1, nrof_centers_batch])
    #similarity_self_mn = tf.ones_like(similarity_all)
    similarity_all_beta = similarity_all + beta
    pre_loss_mtr = tf.subtract(similarity_all_beta, similarity_self_mn)

    ## ignore the element in loss_mtr less than 0, it means the (similarity_within + beta > similarity_inter ) is already satified.
    zero_mtr = tf.zeros_like(pre_loss_mtr, tf.float32)
    loss_mtr = tf.maximum(pre_loss_mtr, zero_mtr)
    loss_sum = tf.reduce_sum(loss_mtr)
    loss_sum_real = tf.sub(loss_sum, beta*nrof_samples)
    loss_mean = tf.div(loss_sum_real, tf.cast(nrof_samples * (nrof_centers_batch-1), tf.float32))

    ## Adding a regularisation term to loss to make it not equal to zero
    loss_reg = tf.add(loss_mean, 1e-10)

    #return loss_reg, loss_mtr, loss_sum, loss_sum_real, loss_mean, pre_loss_mtr, similarity_self_mn, similarity_self, similarity_all, centers, features, nrof_centers_batch
    #return loss_reg, loss_real_mtr, loss_real_mtr, pre_loss_mtr, similarity_self_mn_beta, similarity_self_mn, similarity_self, similarity_all, centers, centers_norm, features, nrof_centers_batch, self_index, a
    return loss_reg, similarity_all, similarity_self, nrof_centers_batch



def center_inter_loss_python(features, label, alfa, nrof_classes, centers): # python version: very slow, 30 time cost than tf version
    """ center_inter_loss = center_loss/||Xi - centers(0,1,2,...i-1,i+1,i+2,...)||
        mzh 22022017
    """
    # loss calculation
    nrof_features = np.shape(features)[0]
    #dim_feature = np.shape(features)[1]
    centers_batch = gather(centers, label)
    dist_center = np.sum(np.square(features - centers_batch),1)
    loss_center = np.sum(dist_center)
    dist_inter_centers = np.zeros([nrof_features, nrof_classes], dtype=np.float32)
    for i in np.arange(nrof_features):
        dist_inter_centers[i,:] = np.sum(np.square(features[i] - centers),1)
    dist_inter_centers_sum = np.sum(dist_inter_centers,1)
    dist_inter_centers_sum = dist_inter_centers_sum - dist_center
    #loss_inter_centers = np.sum(dist_inter_centers_sum)
    loss_inter_centers = np.sum(dist_inter_centers_sum / nrof_classes)
    loss_inter_centers = np.maximum(1e-5, loss_inter_centers)
    loss = loss_center/loss_inter_centers

    # update centers
    centers_cts = np.zeros(nrof_classes, dtype=np.int32)
    centers_batch = gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    for idx in label:
        centers_cts[idx] += 1
    centers_cts_batch = gather(centers_cts, label)
    centers_cts_batch_reshape = np.reshape(centers_cts_batch, [-1,1])
    diff_mean = diff / centers_cts_batch_reshape
    i = 0
    for idx in label:
        centers[idx,:] -= diff_mean[i,:]
        i += 1

    return loss, centers

def gather(data, label):
    i=0
    if data.ndim == 1:
        data_batch = np.zeros(len(label))
        for idx in label:
            data_batch[i] = data[idx]
            i += 1
    if data.ndim == 2:
        data_batch = np.zeros([len(label), np.shape(data)[1]])
        for idx in label:
            data_batch[i,:] = data[idx,:]
            i += 1
    if data.ndim > 2:
        print('The data of dimension should be less than 3!\n')
        assert(data.ndim < 3)

    return data_batch

# def scatter(data, index):
#     return data_sactter


def get_image_paths_and_labels_sfew(images_path, labels_expression):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []
    subs = []
    sub_imgs = []
    idx_train = []
    idx_test = []

    idx_train_sub_all = []
    idx_test_sub_all = []

    with open(labels_expression, 'r') as text_file:
        for line in islice(text_file, 1, None):
            [No, expression, label, img] = str.split(line)
            labels_flat.append(int(label))
            image_paths_flat.append(img)

    nrof_classes = len(image_paths_flat)
    return image_paths_flat, labels_flat, nrof_classes


def get_image_paths_and_labels_oulucasia(images_path, labels_expression, usage, nfold, ifold, isaug=True):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []
    subs = []
    sub_imgs = []
    idx_train = []
    idx_test = []

    idx_train_sub_all = []
    idx_test_sub_all = []

    with open(labels_expression, 'r') as text_file:
        for line in islice(text_file, 1, None):
            [No, sub, expression, label, img] = str.split(line)
            labels_flat.append(int(label))
            image_paths_flat.append(img)
            subs.append(sub)


    subjects = list(set(subs))
    nrof_subj = len(subjects)

    for idx_subj, subj in enumerate(subjects):
        sub_imgs.append([])
        for idx_sub, sub in enumerate(subs):
            if sub == subj:
                sub_imgs[idx_subj].append(idx_sub)

    #folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
    folds = KFold(n=nrof_subj, n_folds=nfold, shuffle=False)

    i = 0
    for idx_train_sub, idx_test_sub in folds:
        idx_train_sub_all.append([])
        idx_train_sub_all[i].append(idx_train_sub)
        idx_test_sub_all.append([])
        idx_test_sub_all[i].append(idx_test_sub)
        print('train:', idx_train_sub, 'test', idx_test_sub)
        i += 1

    idx_train_sub = idx_train_sub_all[ifold][0]
    idx_test_sub = idx_test_sub_all[ifold][0]

    image_paths_flat_array = np.asarray(image_paths_flat)
    labels_flat_array = np.asarray(labels_flat)

    if usage == 'Training':
        for idx in idx_train_sub:
            idx_train += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_train]
        labels_flat_array = labels_flat_array[idx_train]

        ### Reduce the number of the samples of the 'neutral' to balance the number of the classes
        if isaug:
            labels_unique = set(labels_flat_array)
            nrof_classes = len(labels_unique)
            idx_labels_expression = []
            for _ in range(nrof_classes):
                idx_labels_expression.append([])

            for i, lab in enumerate(labels_flat_array):
                idx_labels_expression[lab].append(i)
            idx_labels_neutral = random.sample(idx_labels_expression[0], len(idx_labels_expression[1]))
            idx_labels_augmentation = idx_labels_neutral
            for i in range(1, nrof_classes):
                idx_labels_augmentation += idx_labels_expression[i]

            image_paths_flat_array = image_paths_flat_array[idx_labels_augmentation]
            labels_flat_array = labels_flat_array[idx_labels_augmentation]



    if usage == 'Test':
        for idx in idx_test_sub:
            idx_test += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_test]
        labels_flat_array = labels_flat_array[idx_test]

    image_paths_flat = image_paths_flat_array.tolist()
    labels_flat = labels_flat_array.tolist()

    nrof_classes = len(image_paths_flat)
    return image_paths_flat, labels_flat, usage_flat, nrof_classes

def get_image_paths_and_labels_ckplus(images_path, labels_expression, usage, nfold, ifold):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []
    subs = []
    sub_imgs = []
    idx_train = []
    idx_test = []

    idx_train_sub_all = []
    idx_test_sub_all = []


    with open(labels_expression, 'r') as text_file:
        for line in islice(text_file, 1, None):
            [cnt, sub, session, frame, label] = str.split(line)
            labels_flat.append(int(float(label)))
            subs.append(sub)

            img_name = sub+'_'+session+'_'+frame
            img_path = os.path.join(images_path, sub, session, img_name+'.png')
            image_paths_flat.append(img_path)

    subjects = list(set(subs))
    nrof_subj = len(subjects)

    for idx_subj, subj in enumerate(subjects):
        sub_imgs.append([])
        for idx_sub, sub in enumerate(subs):
            if sub == subj:
                sub_imgs[idx_subj].append(idx_sub)

    #folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
    folds = KFold(n=nrof_subj, n_folds=nfold, shuffle=False)

    i = 0
    for idx_train_sub, idx_test_sub in folds:
        idx_train_sub_all.append([])
        idx_train_sub_all[i].append(idx_train_sub)
        idx_test_sub_all.append([])
        idx_test_sub_all[i].append(idx_test_sub)
        print('train:', idx_train_sub, 'test', idx_test_sub)
        i += 1

    idx_train_sub = idx_train_sub_all[ifold][0]
    idx_test_sub = idx_test_sub_all[ifold][0]

    image_paths_flat_array = np.asarray(image_paths_flat)
    labels_flat_array = np.asarray(labels_flat)

    if usage == 'Training':
        for idx in idx_train_sub:
            idx_train += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_train]
        labels_flat_array = labels_flat_array[idx_train]

    if usage == 'Test':
        for idx in idx_test_sub:
            idx_test += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_test]
        labels_flat_array = labels_flat_array[idx_test]

    image_paths_flat = image_paths_flat_array.tolist()
    labels_flat = labels_flat_array.tolist()    

    nrof_classes = len(image_paths_flat)
    return image_paths_flat, labels_flat, usage_flat, nrof_classes

def get_image_paths_and_labels_fer2013(paths, labels_expression, usage, isaug=False):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        images = os.listdir(path_exp)
        images.sort()
        image_paths_flat = [os.path.join(path_exp,image) for image in images]

    with open(labels_expression, 'r') as text_file:
        for line in islice(text_file, 1, None):
            strtmp = str.split(line)
            labels_flat.append(np.int(strtmp[1]))
            usage_flat.append(strtmp[2])


    index = [idx for idx, phrase in enumerate(usage_flat) if phrase == usage]
    image_paths_flat = [im for idx, im in enumerate(image_paths_flat) if idx in index]
    labels_flat = [im for idx, im in enumerate(labels_flat) if idx in index]
    usage_flat = [im for idx, im in enumerate(usage_flat) if idx in index]

    nrof_classes = len(image_paths_flat)
    
    if usage == 'PublicTest' or usage == 'PrivateTest':
	    return image_paths_flat, labels_flat, usage_flat, nrof_classes

    if isaug :
        #################################################################################
        ###### data augmentation: blancing the number of the samples of each class ######
        #################################################################################
        ## compute the histogram of the dataset
        nrof_expressions = len(set(labels_flat))
        idx_expression_classes = []
        for _ in range(nrof_expressions):
            idx_expression_classes.append([])

        for idx, lab in enumerate(labels_flat):
            if usage_flat[idx] == 'Training':
                idx_expression_classes[lab].append(idx)

        expression_classes_lens = np.zeros(nrof_expressions)
        for i in range(nrof_expressions):
            expression_classes_lens[i] = len(idx_expression_classes[i])
        max_expression_classes_len = int(np.max(expression_classes_lens))

        ## Fill the lists of each expression class by the random samples chosen from the original classes
        for i in range(nrof_expressions):
            idx_expression = idx_expression_classes[i]
            num_to_fill = max_expression_classes_len - len(idx_expression)
            if num_to_fill < len(idx_expression):
                gen_rand_samples = random.sample(idx_expression, num_to_fill)
            else:
                gen_rand_samples = list(itertools.chain.from_iterable(itertools.repeat(x, num_to_fill//len(idx_expression)) for x in idx_expression))

            idx_expression_classes[i] += gen_rand_samples

        ## Flatten the 2D list to 1D list
        idx_expression_classes_1D = list(itertools.chain.from_iterable(idx_expression_classes))

        image_paths_flat_aug = [image_paths_flat[i] for i in idx_expression_classes_1D]
        labels_flat_aug = [labels_flat[i] for i in idx_expression_classes_1D]
        usage_flat_aug = [usage_flat[i] for i in idx_expression_classes_1D]
        nrof_classes_aug = len(labels_flat_aug)

        #return image_paths_flat, labels_flat, usage_flat, nrof_classes
        return image_paths_flat_aug, labels_flat_aug, usage_flat_aug, nrof_classes_aug
    else:
        return image_paths_flat, labels_flat, usage_flat, nrof_classes


def get_image_paths_and_labels_expression(dataset, labels_expression):
    image_paths_flat = []
    labels_flat = []
    image_paths = []
    for i in range(len(dataset)):
        image_paths += dataset[i].image_paths

    with open(labels_expression, 'r') as text_file:
        for line in islice(text_file, 1, None):
            strtmp = str.split(line)
            expr_img = strtmp[1]+'_'+strtmp[2]+'_'+strtmp[3]+'.png'
            matching=[img for img in image_paths if expr_img in img]
            if len([matching]) == 1:
                image_paths_flat.append(matching[0])
                labels_flat.append(int(float(strtmp[-1])))
            else:
                 raise ValueError('Find no or more than one image corresponding to the emotion label!')

    return image_paths_flat, labels_flat

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths_and_labels_recog(dataset):
    image_paths_flat = []
    labels_flat = []
    classes_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        classes_flat += [dataset[i].name]
        labels_flat += [i] * len(dataset[i].image_paths)

    return image_paths_flat, labels_flat, classes_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label
  
def random_rotate_image(image):
    #angle = np.random.uniform(low=-10.0, high=10.0)
    angle = np.random.uniform(low=-180.0, high=180.0)
    return misc.imrotate(image, angle, 'bicubic')
  
def read_and_augument_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, 
        random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
        num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        #pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
  
    return image_batch, label_batch
  
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
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def train_ori(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

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
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    variables_averages_op = variable_averages.apply(update_gradient_vars)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, summary, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    print('######## length of update_gradient_vars: %d\n' % len(update_gradient_vars))
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    ### gradient clip for handling the gradient exploding
    gradslist, varslist = zip(*grads)
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
    return train_op, grads, grads_clip

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

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = np.int(image.shape[1]//2) ##python 3 // int division
        sz2 = np.int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def load_data_test(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = cv2.resize(img,(image_size,image_size))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = cv2.resize(img, (image_size, image_size))
        ##img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def load_data_mega(image_paths, do_random_crop, do_random_flip, do_resize, image_size, BBox, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        image = misc.imread(image_paths[i])
        BBox = BBox.astype(int)
	img = image[BBox[i,0]:BBox[i,0]+BBox[i,2],BBox[i,1]:BBox[i,1]+BBox[i,3],:]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        if do_resize:
            img = cv2.resize(img,(image_size,image_size), interpolation=cv2.INTER_NEAREST)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img

    return images

def load_data_facescrub(image_paths, do_random_crop, do_random_flip, do_resize, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        if do_resize:
            img = cv2.resize(img,(image_size,image_size), interpolation=cv2.INTER_NEAREST)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images


def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                # else:
                #     return learning_rate

        return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_huge_dataset(paths, start_n, end_n):
    dataset = []
    classes = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
	for (d_ino, d_off, d_reclen, d_type, d_name) in python_getdents.getdents64(path_exp):
            if d_name=='.' or d_name == '..':
                continue
            classes += [d_name]

        classes.sort()
       	nrof_classes = len(classes)
        if end_n == -1:
            end_n = nrof_classes
        if end_n>nrof_classes:
            raise ValueError('Invalid end_n:%d more than nrof_class:%d'%(end_n,nrof_classes))      
        for i in range(start_n,end_n):
            if(i%1000 == 0):
                print('reading identities: %d/%d\n'%(i,end_n))
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))


    return dataset



def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
    
# def get_model_filenames(model_dir):
#     files = os.listdir(model_dir)
#     meta_files = [s for s in files if s.endswith('.meta')]
#     if len(meta_files)==0:
#         raise ValueError('No meta file found in the model directory (%s)' % model_dir)
#     elif len(meta_files)>1:
#         raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
#     meta_file = meta_files[0]
#     ckpt_file = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
#     return meta_file, ckpt_file

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    #folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(folds):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, fp_idx, fn_idx = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], fp_idx, fn_idx = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        best_threshold = thresholds[best_threshold_index]

        # #### Global evaluation (not n-fold evaluation) for collecting the indices of the False positive/negative  examples  #####
        _, _, acc_total, fp_idx, fn_idx = calculate_accuracy(best_threshold, dist, actual_issame)
        
    
    return tpr, fpr, accuracy, fp_idx, fn_idx, best_threshold 


def calculate_roc_cosine(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    #diff = np.subtract(embeddings1, embeddings2) ###Eucldian l2 distance
    #dist = np.sum(np.square(diff), 1)

    dist_all = spatial.distance.cdist(embeddings1, embeddings2, 'cosine') ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
    dist = dist_all.diagonal()

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fp_idx,fn_idx = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx], fp_idx, fn_idx = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        best_threshold = thresholds[best_threshold_index]

        # #### Global evaluation (not n-fold evaluation) for collecting the indices of the False positive/negative  examples  #####
        _, _, acc_total, fp_idx, fn_idx = calculate_accuracy(best_threshold, dist, actual_issame)
      
    return tpr, fpr, accuracy, fp_idx, fn_idx, best_threshold

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size

    # ####################################  Edit by mzh 11012017   ####################################
    # #### save the false predict samples: the false posivite (fp) or the false negative(fn) #####
    fp_idx = np.logical_and(predict_issame, np.logical_not(actual_issame))
    fn_idx = np.logical_and(np.logical_not(predict_issame), actual_issame)
    # ####################################  Edit by mzh 11012017   ####################################

    return tpr, fpr, acc, fp_idx, fn_idx

def plot_roc(fpr, tpr, label):
    figure = plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.title('Receiver Operating Characteristics')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], 'g--')
    plt.grid(True)
    plt.show()

    return figure
  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    #folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(folds):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, threshold


def calculate_val_cosine(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    dist_all = spatial.distance.cdist(embeddings1, embeddings2, 'cosine') ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
    dist = dist_all.diagonal()

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, threshold

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_same > 0:
	val = float(true_accept) / float(n_same)
    else:
        val = 0
    if n_diff > 0:
        far = float(false_accept) / float(n_diff)
    else:
        far = 0
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
  
    # #  git hash
    # gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    # (stdout, _) = gitproc.communicate()
    # git_hash = stdout.strip()
    #
    # # Get local changes
    # gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    # (stdout, _) = gitproc.communicate()
    # git_diff = stdout.strip()
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        # text_file.write('git hash: %s\n--------------------\n' % git_hash)
        # text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

## get the labels of  the triplet paths for calculating the center loss - mzh edit 31012017
def get_label_triplet(triplet_paths):
    classes = []
    classes_list = []
    labels_triplet = []
    for image_path in triplet_paths:
        str_items=image_path.split('/')
        classes_list.append(str_items[-2])

    classes = list(sorted(set(classes_list), key=classes_list.index))

    for item in classes_list:
        labels_triplet.append(classes.index(item))

    return  labels_triplet

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def class_filter(image_list, label_list, num_imgs_class):
    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()

    idx = [idx for idx, val  in enumerate(label_num)  if val > num_imgs_class]
    label_idx = [label_key[i] for i in idx]
    idx_list = [i for i in range(0,len(label_list)) if label_list[i] in label_idx]
    label_list_new = [label_list[i] for i in idx_list]
    image_list_new = [image_list[i] for i in idx_list]

    #plt.hist(label_num, bins = 'auto')
    return image_list_new, label_list_new

## Select the images for a epoch in which each batch includes at least two different classes and each class has more than one image
def select_batch_images(image_list, label_list, epoch, epoch_size, batch_size, num_classes_batch, num_imgs_class):
    label_epoch = []
    image_epoch = []

    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()
    nrof_examples = len(image_list)
    nrof_examples_per_epoch = epoch_size*batch_size
    j = epoch*nrof_examples_per_epoch % nrof_examples

    if j+epoch_size*batch_size>nrof_examples:
        j = random.choice(range(0,nrof_examples-epoch_size*batch_size))

    for i in range(epoch_size):
        print('In select_batch_images, batch %d selecting...\n'%(i))
        label_batch = label_list[j+i*batch_size:j+(i+1)*batch_size]
        image_batch = image_list[j+i*batch_size:j+(i+1)*batch_size]

        label_unique =set(label_batch)
        if(len(label_unique)<num_classes_batch or len(label_unique)> (batch_size/num_imgs_class)):
            if (num_classes_batch > (batch_size/num_imgs_class)):
                raise ValueError('The wanted minumum number of classes in a batch (%d classes) is more than the limit can be assigned (%d classes)' %(num_classes_batch, num_imgs_class))
            label_batch = []
            image_batch = []
            ## re-select the image batch which includes num_classes_batch classes
            nrof_im_each_class = np.int(batch_size/num_classes_batch)
            idx = [idx for idx, val in enumerate(label_num) if val>nrof_im_each_class]
            if (len(idx) < num_classes_batch):
                raise ValueError('No enough classes to chose!')
            idx_select = random.sample(idx, num_classes_batch)
            label_key_select = [label_key[i] for i in idx_select]
            for label in label_key_select:
                start_tmp = label_list.index(label)
                idx_tmp = range(start_tmp,start_tmp+nrof_im_each_class+1)
                label_tmp = [label_list[i] for i in idx_tmp]
                img_tmp = [image_list[i] for i in idx_tmp]
                label_batch += label_tmp
                image_batch += img_tmp

            label_batch = label_batch[0:batch_size]
            image_batch = image_batch[0:batch_size]

        label_epoch += label_batch
        image_epoch += image_batch



    return image_epoch, label_epoch
