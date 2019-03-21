#-------------------------------------------------------------------------------
# Name:        ACCurve
# Purpose:
#
# Author:      yag
#
# Created:     04/19/2017
# Copyright:   (c) yag 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from numpy  import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
import argparse
import collections

ACResult = collections.namedtuple('ACResults', ['fig', 'coverage', 'threshold_select']);

def ACCurve(result, style, predIndex, confIndex, pthresh):
    recognitionResults = open(result, 'r');
    resultList= [];
    TotalLabelCnt = 0;

    for imagePairWiseResults in recognitionResults:
        results = imagePairWiseResults.strip('\n').split('\t')
        label = '0';
        pred= results[predIndex];
        groundtruth = results[0];     
        if pred == groundtruth:
            label = '1';
        else: 
            label = '-1';   
        predict = float(results[confIndex]);
        weight = 1.0;
        resultList.append((predict, label, weight))
        TotalLabelCnt = TotalLabelCnt + weight;

    resultList.sort(key = lambda x: x[0], reverse=True)

    tp  = fp  = 0
    coverageList = []
    precisionList = []
    thresholdList =[]

    for result in resultList:
        predict = result[0]
        label = result[1]
        weight = result[2]
        if label == '1':
            tp += weight
        if label == '-1':
            fp += weight

        precision = float(tp) / float(tp+fp)
        coverage = float(tp+fp) / TotalLabelCnt; 

        coverageList.append(coverage)
        precisionList.append(precision)
        thresholdList.append(predict)

    coverage = 0;
    threshold_select = 0;
    for i in range(len(coverageList)-1, 0-1, -1):
        if precisionList[i] > pthresh:
            coverage = coverageList[i];
            threshold_select = thresholdList[i]
            break;

    if(len(coverageList) > 100):
        stepsize = int(floor(len(coverageList)/100));
    else:
        stepsize = 1;

    coverageV = coverageList[0:len(coverageList):stepsize];
    coverageV.append(coverageList[len(coverageList)-1]);
    accuracyV = precisionList[0:len(precisionList):stepsize];
    accuracyV.append(precisionList[len(precisionList)-1]);

    figure, =plt.plot(coverageV, accuracyV, style)
    
    acr = ACResult(figure, coverage, threshold_select);
    return acr


def PCCurveGeneration(args):
    configFile = args.config
    pthresh = float(args.pthresh)
    with open(configFile) as f:
        linesdeduped = sorted(set(line.strip('\n') for line in f))
    curves = []
    legends = []

    for lines in linesdeduped:
        tokens = lines.split('\t')
        predictionEvaluation = tokens[0];
        legend = tokens[1];
        marks = tokens[2];
        predCol =int(tokens[3]);
        confiCol =int(tokens[4]);

        acr = ACCurve(path.normpath(predictionEvaluation), marks, predCol, confiCol, pthresh);
        curves.append(acr.fig);
        if args.coverage == 'yes' and args.threshold == 'yes':
            legends.append(legend+'; C@P=' + str(round(pthresh,4))+': ' + str(round(acr.coverage,4))+', T='+str(round(acr.threshold_select, 4)));
        elif args.coverage == 'yes':
            legends.append(legend+'; Coverage@P=' + str(round(pthresh, 4))+': ' + str(round(acr.coverage,4)));
        elif args.threshold == 'yes':
            legends.append(legend+'; C@P=' + str(round(pthresh, 4))+': ' + str(round(acr.coverage, 4)));
        else:
            legends.append(legend);

    plt.legend(curves, legends, loc=0, prop={'size':12})
    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Coverage', fontsize=16)
    plt.axis([0.0, 1.05, 0.0, 1.05])
    plt.grid(True)
    plt.show()
    color = 'blue';

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', required=True, help = 'input configuration file')
    parser.add_argument('-pthresh', default="0.95", help = 'accuracy threshold to estimate the coverage')
    parser.add_argument('-coverage', default="yes", help = 'print out coverage for the given accuracy on the legend')
    parser.add_argument('-threshold', default="no", help = 'print out threshold for the given accuracy on the legend')
    args = parser.parse_args()
    PCCurveGeneration(args)	

if __name__ == '__main__':
    main()