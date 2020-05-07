from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
sys.path.append('../')
import os
import argparse
#import tensorflow as tf
import numpy as np
#import facenet
#import align.detect_face
import random
from shutil import copyfile
import facenet


### CK+  ###
#EXPRSSIONS_TYPE =  ['0=neutral', '1=anger', '2=contempt', '3=disgust', '4=fear', '5=happy', '6=sadness', '7=surprise']
## OULU-CASIA
#### using 7 expressions including neutral to generate the test dataset, however this won't affect the
##result, howover it needs to map the label of 7 expressions label of 6 expressioin in the generated pairs to adapt the
### trained model
EXPRSSIONS_TYPE_6 =  ['0=Anger', '1=Disgust', '2=Fear', '3=Happiness', '4=Sadness', '5=Surprise' ]
EXPRSSIONS_TYPE =  ['0=Neutral', '1=Anger', '2=Disgust', '3=Fear', '4=Happiness', '5=Sadness', '6=Surprise' ]

def main():
    pairs = []

    data_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160'
    ###labels_expression = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong_Six.txt'
    labels_expression = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
    nfold = 10
    ifold = 4

    #testdata = '/data/zming/datasets/CK+/IdentitySplit_3th_10fold.npy'
    paris_oulucasia = '/data/zming/datasets/Oulu-Casia/IdentitySplit_4th_10fold_oulucasiapairs_Six.txt'

    ##########################   OULU-CASIA    ##########################
    image_paths_test, label_list_test, usage_list_test, nrof_classes_test = facenet.get_image_paths_and_labels_oulucasia(
        data_dir, labels_expression, 'Test', nfold, ifold)

    image_paths_test_np = np.asarray(image_paths_test)
    label_list_test_np = np.asarray(label_list_test)
    #[image_paths_test_np, label_list_test_np] = np.load(testdata)
    ref_expression = [0]
    require_expressions = [4, 6]
    expre_neut = 0

    num_to_pair = 1




    ####### group the images according to the subject including the expression in each tuple ################
    sub_expre_matrix = []
    sublist = []
    for i, im_expr in enumerate(image_paths_test_np):
        sublist.append(str.split(im_expr, '/')[6])

    subs = list(set(sublist))
    for _ in subs:
        sub_expre_matrix.append([])

    for i, im_expr in enumerate(image_paths_test_np):
            sub = str.split(im_expr, '/')[6]
            label = label_list_test_np[i]
            sub_expre_matrix[subs.index(sub)].append([im_expr, label])

    ####### group the images according to the expression including the subject in each tuple ################
    expre_sub_matrix = []
    for _ in EXPRSSIONS_TYPE:
        expre_sub_matrix.append([])

    for i, im_expr in enumerate(image_paths_test_np):
            sub = str.split(im_expr, '/')[6]
            label = label_list_test_np[i]
            expre_sub_matrix[int(label)].append([im_expr, sub])

    #### Select the reference images: ref expression-0=neutral for each subject ###############################
    img_expr_ref = []
    for sub in subs:
        expref = ref_expression[0]
        sub_imgs_exprs = sub_expre_matrix[subs.index(sub)]
        pre_images_expr = [img_expr for img_expr in sub_imgs_exprs if expref in img_expr]
        img_expr_ref += pre_images_expr

    #####################   positive pairs: Same person, Required expressions:'5=happy', '7=surprise' ################:
    for i, im_expr in enumerate(img_expr_ref):
        expr_ref = im_expr[1]
        sub = str.split(im_expr[0], '/')[6]
        sub_imgs_exprs = sub_expre_matrix[subs.index(sub)]
        for expre_require  in require_expressions:
            pre_images_expr = [img_expr for img_expr in sub_imgs_exprs if expre_require in img_expr]
            if pre_images_expr == []:
                continue
            img_expr_selects = random.sample(pre_images_expr, num_to_pair)
            for img_expr_select in img_expr_selects:
                pairs.append([img_expr_select[0], im_expr[0], 'True', img_expr_select[1], expre_require,
                         expr_ref, 'True'])


    ############################### negative paris ####################################### :
    ## same person+not required expression
    for i, im_expr in enumerate(img_expr_ref):
        expr_ref = im_expr[1]
        sub = str.split(im_expr[0], '/')[6]
        sub_imgs_exprs = sub_expre_matrix[subs.index(sub)]
        for expre_require in require_expressions:
            pre_images_expr = [img_expr for img_expr in sub_imgs_exprs if expre_require not in img_expr and expre_neut not in img_expr]
            if pre_images_expr == []:
                continue
            img_expr_selects = random.sample(pre_images_expr, num_to_pair)
            for img_expr_select in img_expr_selects:
                pairs.append([img_expr_select[0], im_expr[0], 'True', img_expr_select[1], expre_require,
                              expr_ref, 'False'])

    ## different person + required expresson
    for i, im_expr in enumerate(img_expr_ref):
        expr_ref = im_expr[1]
        sub = str.split(im_expr[0], '/')[6]
        for expre_require in require_expressions:
            expre_imgs_subs = expre_sub_matrix[int(expre_require)]
            pre_images_expr = [img_expr for img_expr in expre_imgs_subs if sub not in img_expr]
            if pre_images_expr == []:
                continue
            img_expr_selects = random.sample(pre_images_expr, num_to_pair)
            for img_expr_select in img_expr_selects:
                pairs.append([img_expr_select[0], im_expr[0], 'False', expre_require, expre_require,
                              expr_ref, 'True'])

    ## different person + not required expression
    for i, im_expr in enumerate(img_expr_ref):
        expr_ref = im_expr[1]
        sub = str.split(im_expr[0], '/')[6]
        subs_nosub = [subi for subi in subs if subi != sub]
        sub_choice = random.sample(subs_nosub, 1)
        sub_imgs_exprs = sub_expre_matrix[subs.index(sub_choice[0])]

        for expre_require in require_expressions:
            pre_images_expr = [img_expr for img_expr in sub_imgs_exprs if expre_require not in img_expr and expre_neut not in img_expr]
            if pre_images_expr == []:
                continue
            img_expr_selects = random.sample(pre_images_expr, num_to_pair)
            for img_expr_select in img_expr_selects:
                pairs.append([img_expr_select[0], im_expr[0], 'False', img_expr_select[1], expre_require,
                              expr_ref, 'False'])


    ##### map the pairs to the 6 expression system from the current 7 expressions
    for pair in pairs:
        expression_actual = pair[3]
        expression_require = pair[4]
        expression_ref = pair[5]

        expression_actual_6 = [i for i,expr in enumerate(EXPRSSIONS_TYPE_6) if EXPRSSIONS_TYPE[expression_actual][2:] in expr][0]
        expression_require_6 = [i for i, expr in enumerate(EXPRSSIONS_TYPE_6) if EXPRSSIONS_TYPE[expression_require][2:] in expr][0]

        pair[3] = expression_actual_6
        pair[4] = expression_require_6
        pair[5] = -1

    ## write to file
    with open(paris_oulucasia,'wt') as f:
        f.write('No.  Input_img    Ref_img    IdentityIssame     Expression_actual    Expression_require    Expression_ref    ExpressionIsrequire\n')
        for i, line in enumerate(pairs):
            f.write('%d  %s  %s  %s  %s  %s  %s  %s\n' % (i, line[0], line[1], line[2], line[3], line[4], line[5], line[6]))

    return 0



if __name__ == '__main__':
    main()
