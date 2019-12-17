# -*- coding: utf-8 -*-
###############################################################################
# Name        : DataGenerator
# Description : Data generators useable by Keras. It contains several loop
#               closing data generators. Each one is explained in its own
#               comments.
# Note        : See example_datagenerator.py to understand how they work.
#               Nevertheless, the data generators are prepared to be used
#               by fit_generator in Keras, not to be used standalone.
# Note        : Yes, inheritance and polymorphism and all kind of OO stuff
#               could have been used. I know.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 27-June-2019 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from keras.utils import Sequence
import numpy as np  
from skimage.transform import resize
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import cv2
from skimage.feature import hog
from sklearn.cluster import KMeans
from os.path import exists

###############################################################################
# Data generator that builds synthetic loops from the images in a datase and
# extracts HOG features.
# Operation : For each image in the dataset, it builds another image that
#             synthetically closes a loop. This is achieved by translating,
#             rotating and scaling randomly the image.
#             One of the images (original or modified) is randomly selected
#             and returned as it is (only scaled to a specific size).
#             The other image is processed and HOG descriptors are obtained.
#             The output of the generator is a batch where the data info is
#             the image and the HOG descriptors.
# Note      : The size of the HOG descriptors depends on the image size and
#             the specific HOG configuration. If needed to define a Keras model
#             it has to be determined experimentally (i.e. the object does not
#             provide any method to compute it).
###############################################################################
class DataGeneratorHOGImages(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),scaleMin=0.9,scaleMax=1.1,rotateMin=-90,rotateMax=90,
                 txMin=-0.25,txMax=0.25,tyMin=-0.25,tyMax=0.25):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.scaleMin=scaleMin
        self.scaleMax=scaleMax
        self.rotateMin=rotateMin
        self.rotateMax=rotateMax
        self.txMin=txMin
        self.txMax=txMax
        self.tyMin=tyMin
        self.tyMax=tyMax

    def __len__(self):
        return int(np.ceil(self.dataSet.numImages/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numImages)
        for i in range(bStart,bEnd):
            firstImage=self.dataSet.get_image(i)
            secondImage=self.__random_transform__(firstImage)
            theImage,theFeatures=self.__gettuple__([firstImage,secondImage])
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)

    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxHog=int(np.round(np.random.uniform()))
        idxImage=1-idxHog
        hogFeatures=hog(theImages[idxHog],orientations=8,pixels_per_cell=(10,10),cells_per_block=(1, 1),multichannel=True)
        return theImages[idxImage],hogFeatures

    def __image_transform__(self,theImage,scaleFactor,rotationAngle,txFactor,tyFactor):
        centerY,centerX=np.array(theImage.shape[:2])/2.
        theRotation=SimilarityTransform(rotation=np.deg2rad(rotationAngle))
        theZoom=SimilarityTransform(scale=scaleFactor)
        theShift=SimilarityTransform(translation=[-centerX,-centerY])
        theShiftInv=SimilarityTransform(translation=[centerX,centerY])
        theTranslation=SimilarityTransform(translation=[txFactor*2*centerX,tyFactor*2*centerY])
        return warp(theImage, (theShift+(theRotation+theShiftInv))+(theShift+(theZoom+theShiftInv))+theTranslation, mode='reflect')

    def __random_transform__(self,theImage):
        scaleFactor=np.random.uniform(self.scaleMin,self.scaleMax)
        rotationAngle=np.random.uniform(self.rotateMin,self.rotateMax)
        txFactor=np.random.uniform(self.txMin,self.txMax)
        tyFactor=np.random.uniform(self.tyMin,self.tyMax)
        return self.__image_transform__(theImage,scaleFactor,rotationAngle,txFactor,tyFactor)

###############################################################################
# Data generator that uses the existing loops in the database and uses
# HOG descriptors.
# Operation : Same as DataGeneratorHOGImages except that loops are not built
#             synthetically but they come from the dataset itself.
###############################################################################
class DataGeneratorHOGLoops(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320)):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize

    def __len__(self):
        return int(np.ceil(self.dataSet.numLoops/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numLoops)
        imagePairs=[self.dataSet.get_loop(i) for i in range(bStart,bEnd)]
        for curPair in imagePairs:
            theImage,theFeatures=self.__gettuple__(curPair)
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)

    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxHog=int(np.round(np.random.uniform()))
        idxImage=1-idxHog
        hogFeatures=hog(theImages[idxHog],orientations=8,pixels_per_cell=(10,10),cells_per_block=(1, 1),multichannel=True)
        return theImages[idxImage],hogFeatures

###############################################################################
# Data generator that builds synthetic loops from the images in a datase and
# extracts SIFT features.
# Operation : Same as DataGeneratorHOGImages except that SIFT features are
#             used instead of HOG. As for SIFT, the obtained SIFT descriptors
#             are clustered (KMeans) into numDesc groups. The resulting
#             centroids are sorted by the number of associated descriptors in
#             descending order and the first numDesc are selected. If there
#             are not numDesc descriptors, the non-existing are set to zero.
###############################################################################
class DataGeneratorSIFTImages(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),scaleMin=0.9,scaleMax=1.1,rotateMin=-90,rotateMax=90,
                 txMin=-0.25,txMax=0.25,tyMin=-0.25,tyMax=0.25,numDesc=128):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.scaleMin=scaleMin
        self.scaleMax=scaleMax
        self.rotateMin=rotateMin
        self.rotateMax=rotateMax
        self.txMin=txMin
        self.txMax=txMax
        self.tyMin=tyMin
        self.tyMax=tyMax
        self.numDesc=numDesc
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numImages/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numImages)
        for i in range(bStart,bEnd):
            firstImage=self.dataSet.get_image(i)
            secondImage=self.__random_transform__(firstImage)
            theImage,theFeatures=self.__gettuple__([firstImage,secondImage])
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)

    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0],self.imgSize),resize(theLoop[1],self.imgSize))
        idxSIFT=int(np.round(np.random.uniform()))
        idxImage=1-idxSIFT
        siftFeatures=self.__get_descriptors__(theImages[idxSIFT])
        return theImages[idxImage],siftFeatures

    def __image_transform__(self,theImage,scaleFactor,rotationAngle,txFactor,tyFactor):
        centerY,centerX=np.array(theImage.shape[:2])/2.
        theRotation=SimilarityTransform(rotation=np.deg2rad(rotationAngle))
        theZoom=SimilarityTransform(scale=scaleFactor)
        theShift=SimilarityTransform(translation=[-centerX,-centerY])
        theShiftInv=SimilarityTransform(translation=[centerX,centerY])
        theTranslation=SimilarityTransform(translation=[txFactor*2*centerX,tyFactor*2*centerY])
        return warp(theImage, (theShift+(theRotation+theShiftInv))+(theShift+(theZoom+theShiftInv))+theTranslation, mode='reflect')

    def __random_transform__(self,theImage):
        scaleFactor=np.random.uniform(self.scaleMin,self.scaleMax)
        rotationAngle=np.random.uniform(self.rotateMin,self.rotateMax)
        txFactor=np.random.uniform(self.txMin,self.txMax)
        tyFactor=np.random.uniform(self.tyMin,self.tyMax)
        outImage=self.__image_transform__(theImage,scaleFactor,rotationAngle,txFactor,tyFactor)
        return outImage

    def __get_descriptors__(self,theImage):
        outData=np.zeros((self.numDesc,128))
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY)
        theSIFT=cv2.xfeatures2d.SIFT_create()
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None)

        nClust=min(self.numDesc,theDescriptors.shape[0])
        k=KMeans(n_clusters=nClust,random_state=0).fit(theDescriptors)
        idxSort=np.argsort(np.histogram(k.labels_,nClust)[0])[::-1]
        theDescriptors=k.cluster_centers_[idxSort,:]

        dMin=theDescriptors.min(axis=0)
        dMax=theDescriptors.max(axis=0)
        theDescriptors=(theDescriptors-dMin)/(dMax-dMin)
        outData[:theDescriptors.shape[0],:]=theDescriptors
        outData=outData.ravel()

        return outData

    def on_epoch_end(self):
        np.random.seed(0)

###############################################################################
# Data generator that uses the existing loops in the database and uses
# SIFT descriptors.
# Operation : Same as DataGeneratorSIFTImages except that loops are not built
#             synthetically but they come from the dataset itself.
###############################################################################
class DataGeneratorSIFTLoops(Sequence):
    def __init__(self,dataSet,batchSize=20,imgSize=(240,320),numDesc=128):
        self.dataSet=dataSet
        self.batchSize=batchSize
        self.imgSize=imgSize
        self.numDesc=numDesc
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.dataSet.numLoops/float(self.batchSize)))

    def __getitem__(self,theIndex):
        X=[]
        y=[]
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.dataSet.numLoops)
        imagePairs=[[self.dataSet.get_loop(i),i] for i in range(bStart,bEnd)]
        for curPair in imagePairs:
            theImage,theFeatures=self.__gettuple__(curPair)
            X.append(theImage)
            y.append(theFeatures)
        return np.array(X),np.array(y)

    def __gettuple__(self,theLoop):
        theImages=(resize(theLoop[0][0],self.imgSize),resize(theLoop[0][1],self.imgSize))
        idxSIFT=int(np.round(np.random.uniform()))
        idxImage=1-idxSIFT
        siftFeatures=self.__get_descriptors__(theImages[idxSIFT])
        return theImages[idxImage],siftFeatures

    def __get_descriptors__(self,theImage):
        outData=np.zeros((self.numDesc,128))
        # This norm. is just to prevent slightly larger than one values
        theImage/=np.max(theImage)
        ubImage=(theImage*255).astype('uint8')
        gsImage=cv2.cvtColor(ubImage,cv2.COLOR_RGB2GRAY)
        theSIFT=cv2.xfeatures2d.SIFT_create()
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None)

        nClust=min(self.numDesc,theDescriptors.shape[0])
        k=KMeans(n_clusters=nClust,random_state=0).fit(theDescriptors)
        idxSort=np.argsort(np.histogram(k.labels_,nClust)[0])[::-1]
        theDescriptors=k.cluster_centers_[idxSort,:]

        dMin=theDescriptors.min(axis=0)
        dMax=theDescriptors.max(axis=0)
        theDescriptors=(theDescriptors-dMin)/(dMax-dMin)
        outData[:theDescriptors.shape[0],:]=theDescriptors
        outData=outData.ravel()

        return outData


    def on_epoch_end(self):
        np.random.seed(0)