"""Helper for evaluation on the Labeled Faces in the Wild dataset 
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import facenet_ext

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    # tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
    #     np.asarray(actual_issame), nrof_folds=nrof_folds)
    # thresholds = np.arange(0, 4, 0.001)
    # val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
    #     np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    tpr, fpr, accuracy, fp_idx, fn_idx, best_threshold_acc = facenet_ext.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, threshold_val = facenet_ext.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    val_acc, val_std_acc, far_acc, threshold_val_acc = facenet_ext.calculate_val([best_threshold_acc], embeddings1, embeddings2,
                                                                 np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, val, val_std, far, fp_idx, fn_idx, best_threshold_acc, threshold_val, val_acc, far_acc

def evaluate_cosine(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy, fp_idx, fn_idx, best_threshold_acc = facenet.calculate_roc_cosine(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, threshold_val = facenet.calculate_val_cosine(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)


    return tpr, fpr, accuracy, val, val_std, far, fp_idx, fn_idx, best_threshold_acc, threshold_val

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    dataset = str.split(lfw_dir, '/')
    if dataset[4] == 'youtubefacesdb':
        for pair in pairs:
            if len(pair) == 3:
                vid0_dir = str.split(pair[1], '.')
                vid0_dir = vid0_dir[0]
                vid1_dir = str.split(pair[2], '.')
                vid1_dir = vid1_dir[0]
                path0 = os.path.join(lfw_dir, pair[0], vid0_dir, pair[1])
                path1 = os.path.join(lfw_dir, pair[0], vid1_dir, pair[2])
                issame = True
            elif len(pair) == 4:
                vid0_dir = str.split(pair[1], '.')
                vid0_dir = vid0_dir[0]
                vid1_dir = str.split(pair[3], '.')
                vid1_dir = vid1_dir[0]
                path0 = os.path.join(lfw_dir, pair[0], vid0_dir, pair[1])
                path1 = os.path.join(lfw_dir, pair[2], vid1_dir, pair[3])
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list += (path0, path1)
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
    else:
        for pair in pairs:
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list += (path0, path1)
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def get_expr_paths(pairs):
    paths = []
    actual_issame = []
    for pair in pairs:
        #[cnt, img, img_ref, issame, expr_actual, expr_require, expr_ref, expre_isrequire] = str.split(pair)
        img=pair[1]
        img_ref = pair[2]
        issame = pair[3]
        paths.append(img)
        paths.append(img_ref)
        issame = True if issame == 'True' else False
        actual_issame.append(issame)

    return  paths, actual_issame

