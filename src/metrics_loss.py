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
    loss_n = tf.reduce_sum(tf.square(features - centers_batch)/2, 1)
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
    return loss, loss_n, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean # facenet_expression_addcnns_simple_joint_v4_dynamic.py
    #return loss, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean ### facenet_train_classifier_expression_pretrainExpr_multidata_addcnns_simple.py

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


















  


  







  










  






















