# -*- coding: utf-8 -*-
###############################################################################
# Name        : main_trainer
# Description : Performs training, validation and testing using the desired
#               data generator. All combinations of three datasets are used.
# Notes       : To properly use it, proceed as follows:
#               * Consistently define outputSize,theDataGenerator and
#                 savePrefixName
#               * Use a sufficiently large number of epochs (fourth value
#                 in each of the theIndexes items). For example, 100.
#               * Cross your fingers and run the script.
#               * Depending on the number of epochs and the generator, go to:
#                 - Take a cup of coffee.
#                 - Go out and have a beer.
#                 - Go to sleep and check results tomorrow.
#                 - Go on holyday for a couple of days.
#               * Once finished, load the models one by one and plot the
#                 training histories (plot_training_history in ModelWrapper).
#               * By observing the plots, decide the optimal number of epochs.
#               * Change the number of epochs in theIndexes for each case
#                 to the optimal number of epochs observed in previous step.
#               * Change the name of the file where global results are
#                 stored (the one ending with AUC.pkl) to another one. For
#                 example, make it end with AUC2.pkl
#               * Execute the script again, and wait (coffee, sleep, holydays)
#               * When finished, load both the first global results file
#                 (the one ending with AUC.pkl) and the second one (AUC2.pkl)
#               * Compare the values in both files. If values in xxxAUC2.pkl
#                 are all similar or larger than those in xxxAUC.pkl, that's
#                 it: you have good trained models. If some values in xxxAUC2
#                 are significantly smaller than those in xxxAUC... increase
#                 the number of epochs in these cases, and repeat the process
#                 only for those cases (so, just rewrite theIndexes to repre-
#                 sent only the cases you want to modify).
#               * Repeat the process (test, increase epochs...) until
#                 all values in xxxAUC2 are similar or larger than those
#                 in xxxAUC.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

import datagenerator as dg
from dataset import DataSet
from modelwrapper import ModelWrapper
from tester import Tester
from pickle import dump

# Define some parameters
numSIFTDesc=128
hogOutSize=6144
siftOutSize=numSIFTDesc*128
imgSize=(240,320)

# The following three parameters must be consistent
outputSize=hogOutSize
theDataGenerator=dg.DataGeneratorHOGLoops
savePrefixFName='TRAINED_MODELS/HOGLOOPS_'

# Load the datasets and prepare data generators
dataSets=[DataSet('DATASETS/DATASET1.TXT'),DataSet('DATASETS/DATASET2.TXT'),DataSet('DATASETS/DATASET3.TXT')]
dataGenerators=[theDataGenerator(ds,imgSize=imgSize) for ds in dataSets]

# Define all possible combinations (train/validation/test). Last number is the
# number of epochs.
theIndexes=[[0,1,2,10],[0,2,1,10],[1,0,2,10],[1,2,0,10],[2,0,1,10],[2,1,0,10]]

# Prepare storage
allAUC=[]

for curIndexes in theIndexes:
    print('[TRAIN '+str(curIndexes[0]+1)+' VAL '+str(curIndexes[1]+1)+' TEST '+str(curIndexes[2]+1)+']')
    trainGenerator=dataGenerators[curIndexes[0]]
    valGenerator=dataGenerators[curIndexes[1]]
    testGenerator=dataGenerators[curIndexes[2]]

    print('  * TRAINING ')
    theModel=ModelWrapper(inputShape=(imgSize[0],imgSize[1],3),outputSize=outputSize)
    theModel.create()
    theModel.train(trainGenerator,valGenerator,curIndexes[3])

    fileName=savePrefixFName+'T'+str(curIndexes[0]+1)+'_V'+str(curIndexes[1]+1)+'_EPOCHS'+str(curIndexes[3])
    print('  * SAVING MODEL '+fileName)
    theModel.save(fileName)

    print('  * TESTING MODEL')
    theTester=Tester(theModel,dataSets[curIndexes[2]])
    stuff,denseAUC=theTester.compute_hitratio_evolution(useCNN=False)
    stuff,cnnAUC=theTester.compute_hitratio_evolution(useCNN=True)
    print('    + DENSE AUC: '+str(denseAUC))
    print('    + CNN AUC: '+str(cnnAUC))
    allAUC.append([denseAUC,cnnAUC])
    print('  * SAVING TEST RESULTS')
    with open(savePrefixFName+'AUC.pkl','wb') as aucFile:
        dump(allAUC,aucFile)
    print('[DONE]')