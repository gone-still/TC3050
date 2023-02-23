# File        :   main.py (Classifier workflow example)
# Version     :   1.1.5
# Description :   Script that  shows a classic machine learning classifier
#                 workflow implementing a SVM for shape classification
# Date:       :   May 19, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


# Imports:
import numpy as np
import cv2
import random
import os

# import helper functions:
import imageUtils


# Compute shape attributes:
def computeAttributes(sampleList, features):
    # Get list dimensions:
    numberOfSamples = len(sampleList)

    # Prepare the out array:
    outArray = np.zeros((numberOfSamples, features + 1), dtype=np.float32)

    # Attribute computation:
    for i in range(numberOfSamples):
        # Get tuple from list:
        currentTuple = sampleList[i]
        # Get class (string):
        currentClass = currentTuple[0]
        # Get class (number):
        currentClassNumber = currentTuple[1]
        # Get current image:
        currentImage = currentTuple[2]

        # Set class:
        outArray[i][0] = currentClassNumber

        # To Gray:
        grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
        # imageUtils.showImage("Gray", grayImage)

        # Threshold:
        _, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU)
        # imageUtils.showImage("Binary", binaryImage)

        # Get contours:
        contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binaryCopy = binaryImage.copy()
        binaryCopy = cv2.cvtColor(binaryCopy, cv2.COLOR_GRAY2BGR)

        # print("Contours: " + str(len(contours)))

        # Check out the contours, compute attributes:
        for _, c in enumerate(contours):
            # Get the shape's boundig box:
            x, y, w, h = cv2.boundingRect(c)

            # Blob area:
            blobArea = cv2.contourArea(c)

            # Aspect Ratio:
            aspectRatio = w / h

            # Area Difference -> boundingRectArea - blobArea:
            boundingRectArea = w * h
            areaDifference = boundingRectArea - blobArea

            # Circularity:
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * blobArea) / pow(perimeter, 2)

            # Print the info:
            print(currentClass + " A: " + str(aspectRatio) + " D: " + str(areaDifference) + " C: " + str(circularity))

            # Take those 3 attributes and shove them up into the array:
            outArray[i][1] = aspectRatio
            outArray[i][3] = areaDifference
            outArray[i][2] = circularity

            # moments = cv2.moments(im)
            # huMoments = cv2.HuMoments(moments)

    return outArray


# Set image path
path = "D://opencvImages//"
outPath = path + "output//"

# Create the test/train lists:
trainList = []
testList = []

# Roll the PRNG:
random.seed(42)

# Create the class list and dictionary:
shapeClasses = ["circle", "square", "rectangle"]
classesDictionary = {0: "circle", 1: "square", 2: "rectangle"}

# Set the number of train/test samples:
trainSamples = 30
testSamples = 30

# Model flags:
loadModel = False  # Loads model from Hard Drive
saveModel = True  # Saves model to Hard Drive
modelFilename = "svmModel.xml"  # XML model file name

# Get the total shape classes:
totalShapeClasses = len(shapeClasses)
# Create the train data set:
for c in range(totalShapeClasses):
    # Get class as string:
    currentClass = shapeClasses[c]
    # Get class as number 0-2:
    # Implements a "reverse dictionary":
    classNumber = list(classesDictionary.keys())[list(classesDictionary.values()).index(currentClass)]
    for s in range(trainSamples):
        # Create image
        outImage = imageUtils.createShape(currentClass, (200, 320))
        # Into the list:
        trainList.append((currentClass, classNumber, outImage))
        # imageUtils.showImage("Class: " + currentClass + ", Sample: " + str(s), outImage)

# Create the test data set:
for c in range(testSamples):
    # Get random class as number:
    randomClass = random.randint(0, len(shapeClasses) - 1)
    # Get class as string:
    currentClass = classesDictionary[randomClass]

    # Create image
    outImage = imageUtils.createShape(currentClass, (200, 320))
    # Into the list:
    testList.append((currentClass, randomClass, outImage))
    print(currentClass)
    # imageUtils.showImage("Current Shape", outImage)

# Create the train/test matrices:
trainMatrix = computeAttributes(trainList, features=3)
testMatrix = computeAttributes(testList, features=3)

# Get train data and labels:
(trSamples, attributes) = trainMatrix.shape

# Get train labels (first column):
trainLabels = trainMatrix[0:trSamples, 0:1].astype(np.int32)
# Get train data (second to last column):
trainData = trainMatrix[0:trSamples, 1:attributes].astype(np.float32)

# Get test data and labels:
(tsSamples, attributes) = testMatrix.shape

# Get test labels:
testLabels = testMatrix[0:trSamples, 0:1].astype(np.int32)
# Get test data:
testData = testMatrix[0:trSamples, 1:attributes].astype(np.float32)

# Check if the SVM model should be loaded via an XML file
# or we must create the SVM model from scratch:
if not loadModel:

    print("Creating SVM from scratch...")

    # Create the SVM:
    SVM = cv2.ml.SVM_create()

    # Set hyperparameters:
    SVM.setKernel(cv2.ml.SVM_LINEAR)  # Sets the SVM kernel, this is a linear kernel
    SVM.setType(cv2.ml.SVM_NU_SVC)  # Sets the SVM type, this is a "Smooth" Classifier
    SVM.setNu(0.1)  # Sets the "smoothness" of the decision boundary, values: [0.0 - 1.0]

    SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 25, 1.e-01))
    SVM.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

    if saveModel:
        print("Saving SVM XML...")
        modelPath = os.path.join(outPath, modelFilename)
        SVM.save(modelPath)
        print("Saved SVM XML to: " + modelPath)

else:
    print("Loading SVM XML...")
    modelPath = os.path.join(outPath, modelFilename)
    SVM = cv2.ml.SVM_load(modelPath)
    print("Loaded SVM XML from: " + modelPath)

# Test:
# Begin Prediction:
svmResult = SVM.predict(testData)[1]

# Show accuracy
# Create a mask that shows where SVM's prediction matches
# the sample label:
mask = svmResult == testLabels
correct = np.count_nonzero(mask)
print("SVM Accuracy: " + str(correct * 100.0 / svmResult.size) + " %")

# Show each test sample and its classification result:
testImages = testLabels.shape[:1][0]

for s in range(testImages):
    # Get the current test image:
    currentImage = testList[s][2]
    # Get the SVM class for this test image:
    svmPrediction = svmResult[s][0]
    # Get class string based on class number:
    svmLabel = classesDictionary[svmPrediction]

    # Get real class number:
    realClass = testLabels[s][0]
    # Ger real string based on class number:
    realLabel = classesDictionary[realClass]

    # Draw labels on image:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(currentImage, "SVM: " + str(svmLabel), (3, 15), font, 0.5, (0, 255, 0), 1, cv2.LINE_8)
    cv2.putText(currentImage, "TRUTH: " + str(realLabel), (3, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    # Show the classified image:
    imageUtils.showImage("Test Image", currentImage)
