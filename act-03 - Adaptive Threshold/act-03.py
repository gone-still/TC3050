# File        :   act-03.py (Activity 3 of 2022 Spring Vision Course)
# Version     :   1.0.2
# Description :   Introducing adaptive thresholding
# Date:       :   Mar 08, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Read an image:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Image path
path = "D://opencvImages//"
fileName = "sudokuImage.png"

# Reading an image in default mode:
inputImage = readImage(path + fileName)

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Show image:
showImage("grayscaleImage", grayscaleImage)

# Create the histogram
histogram, binLimits = np.histogram(grayscaleImage, bins=256)

# Configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")

# X Axis values:
plt.xlim([0.0, 255.0])

# Plot:
plt.plot(binLimits[0:-1], histogram)
plt.show()

# Try thresholding the image via Otsu:
automaticThreshold, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU)
showImage("binaryImage [Otsu 1]", binaryImage)
# writeImage(path+"sudokuGlobalBinary", binaryImage)

# [Post] Try to clean the image applying
# Gaussian Blur to the grayscale image:
# sigma = (3, 3)
# grayscaleImage = cv2.GaussianBlur(grayscaleImage, sigma, 0)
# showImage("grayscaleImage [Blurred]", grayscaleImage)

# Better ty to threshold local areas (windows) of the image
# Let's apply adaptive thresholding (Gaussian):

windowSize = 7
constantValue = 5
binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize,
                                    constantValue)
# Show the result:
showImage("binaryImage [Adaptive - Gaussian]", binaryImage)
# writeImage(path+"sudokuAdaptative", binaryImage)

# # apply adaptive thresholding (Mean):
# windowSize = 7
# constantValue = 5
# binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, windowSize,
#                                     constantValue)
# # Show the result:
# showImage("binaryImage [Adaptive - Mean]", binaryImage)


# Invert the image:
binaryImage = cv2.subtract(255, binaryImage)
showImage("binaryImage [Inverted]", binaryImage)

# Some noise is still there. We note that the area of the individual "blobs" could be
# use to filter out the noise. Small noise has smaller area than the rest of the blobs
# How can we compute the area of the _individual blobs_ ?
