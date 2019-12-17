# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_modelwrapper
# Description : A basic example on how to use the ModelWrapper class
# Notes       : Just run the script. Be sure that dataset.py is accessible,
#               and the DATASETn.TXT files are accessible, and that the
#               paths specified within DATASETn.TXT are correct.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from dataset import DataSet
from datagenerator import DataGeneratorHOGLoops
from modelwrapper import ModelWrapper

# Load three datasets
print('[[ LOADING DATASETS ]]')
dataSet1=DataSet('DATASETS/DATASET1.TXT')
dataSet2=DataSet('DATASETS/DATASET2.TXT')
dataSet3=DataSet('DATASETS/DATASET3.TXT')
print('[[DATASETS LOADED ]]\n\n')

# Create three data generators
print('[[ CREATING DATA GENERATORS ]]')
dataGenerator1=DataGeneratorHOGLoops(dataSet1)
dataGenerator2=DataGeneratorHOGLoops(dataSet2)
dataGenerator3=DataGeneratorHOGLoops(dataSet3)
print('[[ GENERATORS CREATED ]]\n\n')

# Create the model
print('[[ CREATING THE MODEL ]]')
theModel=ModelWrapper(outputSize=6144)
theModel.create()
print('[[ MODEL CREATED ]]')

# Train the model with dataset1 and validate with dataset2
print('[[ TRAINING WITH DATASET1 AND VALIDATING WITH DATASET2 ]]')
theModel.train(trainGenerator=dataGenerator1,valGenerator=dataGenerator2,nEpochs=10)
print('[[ MODEL TRAINED ]]')

# Save the model
print('[[ SAVING THE MODEL ]]')
theModel.save('TEST_MODEL')
print('[[ MODEL SAVED ]]')

# Loading the model (not necessary, since it is already loaded. Loading is
# performed just for the sake of completeness)
print('[[ LOADING THE MODEL ]]')
theModel.load('TEST_MODEL')
print('[[ MODEL SAVED ]]')

# Plot the training history
print('[[ PLOTTING TRAINING HISTORY ]]')
theModel.plot_training_history()
print('[[ PLOT DONE ]]')

# Evaluating the model with dataset3
print('[[ EVALUATING THE MODEL WITH DATASET3 ]]')
print(theModel.evaluate(dataGenerator3))
print('[[ MODEL EVALUATED ]]')