"""Functions for building the face recognition network.
"""
# MIT License
#### copyright at Auther: mingzuheng, 25/03/2019 #########

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
import sys


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




def label_mapping(label_list_src, EXPRSSIONS_TYPE_src, EXPRSSIONS_TYPE_trg):
    labels_mapping = []
    idx_label_notexist = []
    for i, label in enumerate(label_list_src):
        expre_src = str.split(EXPRSSIONS_TYPE_src[label], '=')[1]
        expre_trg = [x for x in EXPRSSIONS_TYPE_trg if expre_src in x]
        if expre_trg == []:
            label_trg = -1
            idx_label_notexist.append(i)
        else:
            label_trg = int(str.split(expre_trg[0], '=')[0])
        labels_mapping.append(label_trg)

    return idx_label_notexist, labels_mapping


def gather(data, label):
    i = 0
    if data.ndim == 1:
        data_batch = np.zeros(len(label))
        for idx in label:
            data_batch[i] = data[idx]
            i += 1
    if data.ndim == 2:
        data_batch = np.zeros([len(label), np.shape(data)[1]])
        for idx in label:
            data_batch[i, :] = data[idx, :]
            i += 1
    if data.ndim > 2:
        print('The data of dimension should be less than 3!\n')
        assert (data.ndim < 3)

    return data_batch


# def scatter(data, index):
#     return data_sactter

def generate_labels_id(subs):
    subjects = list(set(subs))
    subjects = np.sort(subjects)
    labels_id = []
    for sub in subs:
        labels_id.append([idx for idx, subject in enumerate(subjects) if sub == subject][0])

    return labels_id


def generate_idiap_image_label(path, quality, framenum_dist):
    if quality == 'real':
        label = 1
    elif quality == 'attack':
        label = 0
    else:
        raise ValueError("Invalid quality of the images!")

    img_list = []
    label_list = []
    i = 0
    videos = os.listdir(path)
    videos.sort()
    for video in videos:
        video_path = os.path.join(path, video)
        if os.path.isdir(video_path):
            imgs = os.listdir(video_path)
            imgs.sort()
            imgs_select = random.sample(imgs, framenum_dist)
            for img in imgs_select:
                img_path = os.path.join(video_path, img)
                img_list.append(img_path)
                label_list.append(label)
                i += 1
                # print('%d %s %s' % (i, img_path, label))

    return img_list, label_list, i


def get_image_paths_and_labels_idiap(data_dir_dist, framenum_dist_real, framenum_dist_attack):
    images_list = []
    labels_list = []
    cnt_total = 0

    qualities = os.listdir(data_dir_dist)
    for quality in qualities:
        if quality == 'real':
            imgs, labels, cnt = generate_idiap_image_label(os.path.join(data_dir_dist, quality), quality,
                                                           framenum_dist_real)
            images_list += imgs
            labels_list += labels
            cnt_total += cnt

        elif quality == 'attack':
            attack_styles = os.listdir(os.path.join(data_dir_dist, quality))
            for attack_style in attack_styles:
                imgs, labels, cnt = generate_idiap_image_label(os.path.join(data_dir_dist, quality, attack_style),
                                                               quality, framenum_dist_attack)
                images_list += imgs
                labels_list += labels
                cnt_total += cnt
        else:
            raise ValueError("Invalid quality of the images!")

    return images_list, labels_list, cnt_total

def get_image_paths_and_labels_genki4k(labels_expression, usage):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []

    ## read labels
    with open(labels_expression,'r') as f:
        s = f.readlines()
        for line in s:
            x = line[66:-1]
            if  x == usage:
                label = int(line[0])
                labels_flat.append(label)
                image = line[2:65]
                image_paths_flat.append(image)
                usage_flat.append(usage)


    nrof_classes = len(labels_flat)

    return image_paths_flat, labels_flat, usage_flat, nrof_classes

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


def get_image_paths_and_labels_oulucasia(images_path, labels_expression, usage, nfold, ifold, isaug=False):
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

    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
    folds = KFold(n=nrof_subj, n_folds=nfold, shuffle=False)

    i = 0
    for idx_train_sub, idx_test_sub in folds:
        idx_train_sub_all.append([])
        idx_train_sub_all[i].append(idx_train_sub)
        idx_test_sub_all.append([])
        idx_test_sub_all[i].append(idx_test_sub)
        # print('train:', idx_train_sub, 'test', idx_test_sub)
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

    # nrof_classes = len(image_paths_flat)
    nrof_classes = nrof_subj
    return image_paths_flat, labels_flat, usage_flat, nrof_classes


def get_image_paths_and_labels_joint_oulucasia(images_path, labels_expression, usage, nfold, ifold, isaug=True):
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
    labels_id = generate_labels_id(subs)

    for idx_subj, subj in enumerate(subjects):
        sub_imgs.append([])
        for idx_sub, sub in enumerate(subs):
            if sub == subj:
                sub_imgs[idx_subj].append(idx_sub)

    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
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
    labels_id_array = np.asarray(labels_id)

    if usage == 'Training':
        for idx in idx_train_sub:
            idx_train += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_train]
        labels_flat_array = labels_flat_array[idx_train]
        labels_id_array = labels_id_array[idx_train]

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
            labels_id_array = labels_id_array[idx_labels_augmentation]

    if usage == 'Test':
        for idx in idx_test_sub:
            idx_test += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_test]
        labels_flat_array = labels_flat_array[idx_test]
        labels_id_array = labels_id_array[idx_test]

    image_paths_flat = image_paths_flat_array.tolist()
    labels_flat = labels_flat_array.tolist()
    labels_id_flat = labels_id_array.tolist()

    nrof_classes = len(set(labels_id_flat))

    return image_paths_flat, labels_flat, usage_flat, nrof_classes, labels_id_flat


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

            img_name = sub + '_' + session + '_' + frame
            img_path = os.path.join(images_path, sub, session, img_name + '.png')
            image_paths_flat.append(img_path)

    subjects = list(set(subs))
    nrof_subj = len(subjects)

    for idx_subj, subj in enumerate(subjects):
        sub_imgs.append([])
        for idx_sub, sub in enumerate(subs):
            if sub == subj:
                sub_imgs[idx_subj].append(idx_sub)

    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
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


def get_image_paths_and_labels_joint_ckplus(images_path, labels_expression, usage, nfold, ifold):
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

            img_name = sub + '_' + session + '_' + frame
            img_path = os.path.join(images_path, sub, session, img_name + '.png')
            image_paths_flat.append(img_path)

    subjects = list(set(subs))
    nrof_subj = len(subjects)
    labels_id = generate_labels_id(subs)

    for idx_subj, subj in enumerate(subjects):
        sub_imgs.append([])
        for idx_sub, sub in enumerate(subs):
            if sub == subj:
                sub_imgs[idx_subj].append(idx_sub)

    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
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
    labels_id_array = np.asarray(labels_id)

    if usage == 'Training':
        for idx in idx_train_sub:
            idx_train += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_train]
        labels_flat_array = labels_flat_array[idx_train]
        labels_id_array = labels_id_array[idx_train]

    if usage == 'Test':
        for idx in idx_test_sub:
            idx_test += sub_imgs[idx]

        image_paths_flat_array = image_paths_flat_array[idx_test]
        labels_flat_array = labels_flat_array[idx_test]
        labels_id_array = labels_id_array[idx_test]

    image_paths_flat = image_paths_flat_array.tolist()
    labels_flat = labels_flat_array.tolist()
    labels_id_flat = labels_id_array.tolist()

    nrof_classes = len(set(labels_id_flat))

    return image_paths_flat, labels_flat, usage_flat, nrof_classes, labels_id_flat


def get_image_paths_and_labels_fer2013(paths, labels_expression, usage, isaug=False):
    image_paths_flat = []
    labels_flat = []
    usage_flat = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        images = os.listdir(path_exp)
        images.sort()
        image_paths_flat = [os.path.join(path_exp, image) for image in images]

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

    if isaug:
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
                gen_rand_samples = list(itertools.chain.from_iterable(
                    itertools.repeat(x, num_to_fill // len(idx_expression)) for x in idx_expression))

            idx_expression_classes[i] += gen_rand_samples

        ## Flatten the 2D list to 1D list
        idx_expression_classes_1D = list(itertools.chain.from_iterable(idx_expression_classes))

        image_paths_flat_aug = [image_paths_flat[i] for i in idx_expression_classes_1D]
        labels_flat_aug = [labels_flat[i] for i in idx_expression_classes_1D]
        usage_flat_aug = [usage_flat[i] for i in idx_expression_classes_1D]
        nrof_classes_aug = len(labels_flat_aug)

        # return image_paths_flat, labels_flat, usage_flat, nrof_classes
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
            expr_img = strtmp[1] + '_' + strtmp[2] + '_' + strtmp[3] + '.png'
            matching = [img for img in image_paths if expr_img in img]
            if len([matching]) == 1:
                image_paths_flat.append(matching[0])
                labels_flat.append(int(float(strtmp[-1])))
            else:
                raise ValueError('Find no or more than one image corresponding to the emotion label!')

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


def random_rotate_image(image):
    # angle = np.random.uniform(low=-10.0, high=10.0)
    angle = np.random.uniform(low=-180.0, high=180.0)
    return misc.imrotate(image, angle, 'bicubic')


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def crop(image, random_crop, image_size):
    if min(image.shape[0], image.shape[1]) > image_size:
        sz1 = image.shape[0] // 2
        sz2 = image.shape[1] // 2

        crop_size = image_size//2
        diff_h = sz1 - crop_size
        diff_v = sz2 - crop_size
        (h, v) = (np.random.randint(-diff_h, diff_h + 1), np.random.randint(-diff_v, diff_v + 1))

        image = image[(sz1+h-crop_size):(sz1+h+crop_size ), (sz2+v-crop_size):(sz2+v+crop_size ), :]
    # else:
    #     print("Image size is small than crop image size!")

    return image

def load_data_test(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = cv2.resize(img, (image_size, image_size))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = cv2.resize(img, (image_size, image_size))
        ##img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images


def load_data_mega(image_paths, do_random_crop, do_random_flip, do_resize, image_size, BBox, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        image = misc.imread(image_paths[i])
        BBox = BBox.astype(int)
        img = image[BBox[i, 0]:BBox[i, 0] + BBox[i, 2], BBox[i, 1]:BBox[i, 1] + BBox[i, 3], :]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        if do_resize:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img

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
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        if do_random_crop:
            img = crop(img, do_random_crop, image_size)
        if do_random_flip:
            img = flip(img, do_random_flip)

        images[i, :, :, :] = img
    return images


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
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_huge_dataset(paths, start_n=0, end_n=-1):
    dataset = []
    classes = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        for (d_ino, d_off, d_reclen, d_type, d_name) in python_getdents.getdents64(path_exp):
            if d_name == '.' or d_name == '..':
                continue
            classes += [d_name]

        classes.sort()
        nrof_classes = len(classes)
        if end_n == -1:
            end_n = nrof_classes
        if end_n > nrof_classes:
            raise ValueError('Invalid end_n:%d more than nrof_class:%d' % (end_n, nrof_classes))
        for i in range(start_n, end_n):
            if (i % 1000 == 0):
                print('reading identities: %d/%d\n' % (i, end_n))
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True, do_resize=True):
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
        if do_resize:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        images[i,:,:,:] = img

    return images

def load_data_im(imgs, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    # nrof_samples = len(image_paths)
    if (len(imgs.shape) > 3):
        nrof_samples = imgs.shape[0]
    elif (len(imgs.shape) == 3):
        nrof_samples = 1
    elif (len(imgs.shape) == 1):
        nrof_samples = len(imgs)
    else:
        print('No images!')
        return -1

    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        # img = misc.imread(image_paths[i])
        if nrof_samples > 1:
            img = imgs[i]
        else:
            img = imgs

        if len(img):
            if img.ndim == 2:
                img = to_rgb(img)
            if do_prewhiten:
                img = prewhiten(img)
            img = crop(img, do_random_crop, image_size)
            img = flip(img, do_random_flip)
            images[i] = img
    images = np.squeeze(images)
    return images


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_threshold = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set],
                                                                                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_threshold[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fp_idx, fn_idx = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], fp_idx, fn_idx = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    mean_best_threshold = np.mean(best_threshold)

    # #### Global evaluation (not n-fold evaluation) for collecting the indices of the False positive/negative  examples  #####
    _, _, acc_total, fp_idx, fn_idx = calculate_accuracy(mean_best_threshold, dist, actual_issame)

    return tpr, fpr, accuracy, fp_idx, fn_idx, mean_best_threshold


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

    # diff = np.subtract(embeddings1, embeddings2) ###Eucldian l2 distance
    # dist = np.sum(np.square(diff), 1)

    dist_all = spatial.distance.cdist(embeddings1, embeddings2,
                                      'cosine')  ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
    dist = dist_all.diagonal()

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set],
                                                                                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fp_idx, fn_idx = calculate_accuracy(
                threshold,
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

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

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
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    for fold_idx, (train_set, test_set) in enumerate(folds):

        if nrof_thresholds > 1:
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0
        else:
            threshold = thresholds[0]

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

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
    dist_all = spatial.distance.cdist(embeddings1, embeddings2,
                                      'cosine')  ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
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


## get the labels of  the triplet paths for calculating the center loss - mzh edit 31012017
def get_label_triplet(triplet_paths):
    classes = []
    classes_list = []
    labels_triplet = []
    for image_path in triplet_paths:
        str_items = image_path.split('/')
        classes_list.append(str_items[-2])

    classes = list(sorted(set(classes_list), key=classes_list.index))

    for item in classes_list:
        labels_triplet.append(classes.index(item))

    return labels_triplet


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def class_filter(image_list, label_list, num_imgs_class):
    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()

    idx = [idx for idx, val in enumerate(label_num) if val > num_imgs_class]
    label_idx = [label_key[i] for i in idx]
    idx_list = [i for i in range(0, len(label_list)) if label_list[i] in label_idx]
    label_list_new = [label_list[i] for i in idx_list]
    image_list_new = [image_list[i] for i in idx_list]

    # plt.hist(label_num, bins = 'auto')
    return image_list_new, label_list_new


## Select the images for a epoch in which each batch includes at least two different classes and each class has more than one image
def select_batch_images(image_list, label_list, epoch, epoch_size, batch_size, num_classes_batch, num_imgs_class):
    label_epoch = []
    image_epoch = []

    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()
    nrof_examples = len(image_list)
    nrof_examples_per_epoch = epoch_size * batch_size
    j = epoch * nrof_examples_per_epoch % nrof_examples

    if j + epoch_size * batch_size > nrof_examples:
        j = random.choice(range(0, nrof_examples - epoch_size * batch_size))

    for i in range(epoch_size):
        print('In select_batch_images, batch %d selecting...\n' % (i))
        label_batch = label_list[j + i * batch_size:j + (i + 1) * batch_size]
        image_batch = image_list[j + i * batch_size:j + (i + 1) * batch_size]

        label_unique = set(label_batch)
        if (len(label_unique) < num_classes_batch or len(label_unique) > (batch_size / num_imgs_class)):
            if (num_classes_batch > (batch_size / num_imgs_class)):
                raise ValueError(
                    'The wanted minumum number of classes in a batch (%d classes) is more than the limit can be assigned (%d classes)' % (
                    num_classes_batch, num_imgs_class))
            label_batch = []
            image_batch = []
            ## re-select the image batch which includes num_classes_batch classes
            nrof_im_each_class = np.int(batch_size / num_classes_batch)
            idx = [idx for idx, val in enumerate(label_num) if val > nrof_im_each_class]
            if (len(idx) < num_classes_batch):
                raise ValueError('No enough classes to chose!')
            idx_select = random.sample(idx, num_classes_batch)
            label_key_select = [label_key[i] for i in idx_select]
            for label in label_key_select:
                start_tmp = label_list.index(label)
                idx_tmp = range(start_tmp, start_tmp + nrof_im_each_class + 1)
                label_tmp = [label_list[i] for i in idx_tmp]
                img_tmp = [image_list[i] for i in idx_tmp]
                label_batch += label_tmp
                image_batch += img_tmp

            label_batch = label_batch[0:batch_size]
            image_batch = image_batch[0:batch_size]

        label_epoch += label_batch
        image_epoch += image_batch

    return image_epoch, label_epoch


def label_mapping(label_list_src, EXPRSSIONS_TYPE_src, EXPRSSIONS_TYPE_trg):
    labels_mapping = []
    idx_label_notexist = []
    for i, label in enumerate(label_list_src):
        expre_src = str.split(EXPRSSIONS_TYPE_src[label], '=')[1]
        expre_trg = [x for x in EXPRSSIONS_TYPE_trg if expre_src in x]
        if expre_trg == []:
            label_trg = -1
            idx_label_notexist.append(i)
        else:
            label_trg = int(str.split(expre_trg[0], '=')[0])
        labels_mapping.append(label_trg)

    return idx_label_notexist, labels_mapping