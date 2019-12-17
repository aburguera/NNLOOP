# Visual Loop Closing using Neural Networks

This code makes it possible to close visual loops using Neural Networks. The process is specifically designed to work with underwater images gathered by a bottom looking camera.

* Science : Antoni Burguera (antoni dot burguera at uib dot es)
            Francisco Bonin-Font
* Coding  : Antoni Burguera (antoni dot burguera at uib dot es)

If you use this software, please cite the following paper:

Paper reference: to be posted soon. Please contact us.

Also, the whole process is carefully described in the paper. Please read the paper to understand how the system works.

## The datasets

This code is provided without the full datasets. A dataset is composed of three parts:

* A set of database images. Not provided.
* A set of query images. Not provided.
* A dataset specifier. Three examples are available in the DATASETS folder.

The dataset specifier is a text file with the following fields separated by #

* Path to the folder with the database images
* Path to the folder with the query images
* File names (relative to the specified path) of the database images, separated by commas
* File names (relative to the specified path) of the query images, separated by commas
* Loop specs as a set of indexes separated by commas. The format is as follows: databaseImage[index[2\*i]] closes a loop with queryImage[index[2\*i+1]], where databaseImage and queryImage are the images as they apper in the previous fields. It is important that ALL the existing loops are specified in this way

Each query image must close at least one loop with one database image. Given one query, all the database images that do not appear in the loop specs for that query must NOT close a loop with that query.

Please note that the loops specified here are the ground truth. So, they are NOT found by the software. Instead, they are used to validate the software.

## Understanding the system

The main modules are:

* DataSet : Loads and manages datasets. It makes easy to access the database, query and loop data.
* DataGenerator : Keras data generators for different ways to feed the Neural Network. Check the paper for more information.
* ModelWrapper : Simple wrapper to ease the creation, loading, saving, ... the Keras model.
* Tester : Tests the system in different ways.
* MotionEstimator : Estimates the relative position between 2D point clouds using RANSAC.
* ImageMatcher : Matches two images using RANSAC.

For each of these modules, there is an usage example available. To understand each of them just check the corresponding example_* file. There is an addition example: example_candidates.py, which shows a simple way to use a pre-trained model to select loop closing candidates.

## Using the system

First of all, prepare your own datasets. Afterwards, you can execute:

* main_trainer : Trains the system and stores the trained models.
* main_tester : Checks the system. Uses the NN to get loop candidates and confirms them using RANSAC. Different stats are obtained too.

Also, after training at least one model you can use it to execute the example in example_candidates.py. Just change the model file name to the one you trained.

## Requirements

To execute this software, you will need:

* Python 3
* Keras
* Tensorflow
* NumPy
* SciKit-Image
* SciKit-Learn

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it. Just be careful and try to understand everything before using it. If you have questions, please carefully read the code and the paper. If this doesn't help, contact us.