# -*- coding: utf-8 -*-
###############################################################################
# Name        : main_tester
# Description : Performs the final testing using trained and validated
#               models.
# Notes       : To properly use it, proceed as follows:
#               * Select a search ratio (ratio of database images to select
#                 to perform RANSAC on them)
#               * Define theIndexes. To compute the files referred in
#                 theIndexes,please check main_trainer. The files used here
#                 correspond to our own experiments and may not work properly
#                 with different data. Feel free to generate your own.
#               * Execute the script.
#               * Depending on the search ratio, the time to completion can be:
#                 - One Cobra Kai episode
#                 - The Lord of the Rings trilogy, director's cut
#                 - All Dr. Who seasons (old and new doctors)
#               * Results are stored periodically. So, if the program stops
#                 for some reason, you can check the last iteration stored
#                 and modify the code to continue there.
#               * Also, once the test is performed you can just analyze the
#                 results from the file by commenting the call to do_all_tests
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
from modelwrapper import ModelWrapper
from tester import Tester
from pickle import dump,load
import numpy as np

# Cases to test. Each item has:
# * First: Model file name, as stored by main_trainer
# * Second: Test dataset
theIndexes=[['TRAINED_MODELS/SIFTIMAGES_T1_V2_EPOCHS15',2],
            ['TRAINED_MODELS/SIFTIMAGES_T1_V3_EPOCHS15',1],
            ['TRAINED_MODELS/SIFTIMAGES_T2_V1_EPOCHS15',2],
            ['TRAINED_MODELS/SIFTIMAGES_T2_V3_EPOCHS15',0],
            ['TRAINED_MODELS/SIFTIMAGES_T3_V1_EPOCHS13',1],
            ['TRAINED_MODELS/SIFTIMAGES_T3_V2_EPOCHS19',0],
            ['TRAINED_MODELS/SIFTLOOPS_T1_V2_EPOCHS20',2],
            ['TRAINED_MODELS/SIFTLOOPS_T1_V3_EPOCHS19',1],
            ['TRAINED_MODELS/SIFTLOOPS_T2_V1_EPOCHS22',2],
            ['TRAINED_MODELS/SIFTLOOPS_T2_V3_EPOCHS20',0],
            ['TRAINED_MODELS/SIFTLOOPS_T3_V1_EPOCHS19',1],
            ['TRAINED_MODELS/SIFTLOOPS_T3_V2_EPOCHS22',0],
            ['TRAINED_MODELS/HOGIMAGES_T1_V2_EPOCHS27',2],
            ['TRAINED_MODELS/HOGIMAGES_T1_V3_EPOCHS27',1],
            ['TRAINED_MODELS/HOGIMAGES_T2_V1_EPOCHS27',2],
            ['TRAINED_MODELS/HOGIMAGES_T2_V3_EPOCHS26',0],
            ['TRAINED_MODELS/HOGIMAGES_T3_V1_EPOCHS25',1],
            ['TRAINED_MODELS/HOGIMAGES_T3_V2_EPOCHS26',0],
            ['TRAINED_MODELS/HOGLOOPS_T1_V2_EPOCHS24',2],
            ['TRAINED_MODELS/HOGLOOPS_T1_V3_EPOCHS23',1],
            ['TRAINED_MODELS/HOGLOOPS_T2_V1_EPOCHS25',2],
            ['TRAINED_MODELS/HOGLOOPS_T2_V3_EPOCHS19',0],
            ['TRAINED_MODELS/HOGLOOPS_T3_V1_EPOCHS21',1],
            ['TRAINED_MODELS/HOGLOOPS_T3_V2_EPOCHS24',0]]

searchRatio=0.1

def do_all_tests(theIndexes,searchRatio):
    dataSets=[DataSet('DATASETS/DATASET1.TXT'),DataSet('DATASETS/DATASET2.TXT'),DataSet('DATASETS/DATASET3.TXT')]
    allStats=[]
    theTester=Tester()
    theModel=ModelWrapper()

    print('[[[[ STARTING THE MOTHER OF ALL TESTS ]]]]')
    for useCNN in [False,True]:
        print('[[[ ONLY CNN LAYERS '+str(useCNN).upper()+' ]]]')
        for curIndex in theIndexes:
            print('[[ TESTING MODEL '+curIndex[0]+' WITH TEST SET '+str(curIndex[1]+1)+' ]]')
            theModel.load(curIndex[0])
            theTester.set_params(theModel,dataSets[curIndex[1]])
            curStats=theTester.compute_fullstats(useCNN=useCNN,searchRatio=searchRatio)
            allStats.append(curStats)
            print('[[ MODEL TESTED ]]')
            with open('ALLSTATS_PCT'+str(int(searchRatio*100))+'.pkl','wb') as outFile:
                dump(allStats,outFile)
        print('[[[ FINISHED ONLY CNN LAYERS '+str(useCNN).upper()+' ]]]')
    print('[[[[ FINISHED THE MOTHER OF ALL TESTS ]]]]')

def get_indexes(theIndexes,subString,useCNN):
    outIndexes=[]
    for i in range(len(theIndexes)):
        if subString in theIndexes[i][0]:
            if useCNN:
                outIndex=i+len(theIndexes)
            else:
                outIndex=i
            outIndexes.append(outIndex)
    return np.array(outIndexes)

def get_stats(theStats,doPrint=False):
    tp=np.sum(theStats[:,0])
    fp=np.sum(theStats[:,1])
    tn=np.sum(theStats[:,2])
    fn=np.sum(theStats[:,3])
    tdist=np.mean(theStats[:,4])
    tloop=np.mean(theStats[:,5])
    theAccuracy=(tp+tn)/(tp+fp+tn+fn)
    theTPR=tp/(tp+fn)
    theFPR=fp/(fp+tn)
    if doPrint:
        print('  * ACCURACY   : '+str(theAccuracy))
        print('  * TPR        : '+str(theTPR))
        print('  * FPR        : '+str(theFPR))
        print('  * TOTAL TIME : '+str(tdist+tloop))
    return theAccuracy,theTPR,theFPR,tdist+tloop

def analyze_results(theIndexes,allStats):
    allStats=np.array(allStats)

    idxSIFTImagesDense=get_indexes(theIndexes,'SIFTIMAGES',False)
    idxSIFTImagesCNN=get_indexes(theIndexes,'SIFTIMAGES',True)
    idxSIFTLoopsDense=get_indexes(theIndexes,'SIFTLOOPS',False)
    idxSIFTLoopsCNN=get_indexes(theIndexes,'SIFTLOOPS',True)

    idxHOGImagesDense=get_indexes(theIndexes,'HOGIMAGES',False)
    idxHOGImagesCNN=get_indexes(theIndexes,'HOGIMAGES',True)
    idxHOGLoopsDense=get_indexes(theIndexes,'HOGLOOPS',False)
    idxHOGLoopsCNN=get_indexes(theIndexes,'HOGLOOPS',True)

    print('[ RESULTS FOR SIFT WITH SYNTHETIC LOOPS AND ALL NN LAYERS ]')
    print(get_stats(allStats[idxSIFTImagesDense,:]))
    print('[ RESULTS FOR SIFT WITH SYNTHETIC LOOPS AND ONLY CONV. LAYERS ]')
    print(get_stats(allStats[idxSIFTImagesCNN,:]))
    print('[ RESULTS FOR SIFT WITH EXISTING LOOPS AND ALL NN LAYERS ]')
    print(get_stats(allStats[idxSIFTLoopsDense,:]))
    print('[ RESULTS FOR SIFT WITH EXISTING LOOPS AND ONLY CONV. LAYERS ]')
    print(get_stats(allStats[idxSIFTLoopsCNN,:]))
    print('[ RESULTS FOR HOG WITH SYNTHETIC LOOPS AND ALL NN LAYERS ]')
    print(get_stats(allStats[idxHOGImagesDense,:]))
    print('[ RESULTS FOR HOG WITH SYNTHETIC LOOPS AND ONLY CONV. LAYERS ]')
    print(get_stats(allStats[idxHOGImagesCNN,:]))
    print('[ RESULTS FOR HOG WITH EXISTING LOOPS AND ALL NN LAYERS ]')
    print(get_stats(allStats[idxHOGLoopsDense,:]))
    print('[ RESULTS FOR HOG WITH EXISTING LOOPS AND ONLY CONV. LAYERS ]')
    print(get_stats(allStats[idxHOGLoopsCNN,:]))


# Do all the tests. It is time and memory consuming. Depending on your memory
# it can hang the computer. If that happens, check what is saved in ALLSTATS*
# file to see where continue.
# Once compute, you can comment the call to do_all_tests to directly see the
# results.
do_all_tests(theIndexes,searchRatio)

with open('ALLSTATS_PCT'+str(int(searchRatio*100))+'.pkl','rb') as inFile:
    allStats=load(inFile)

analyze_results(theIndexes,allStats)
