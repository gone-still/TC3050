# File        :   act-02.py (Activity 2 of 2022 Spring Vision Course)
# Version     :   1.1.1
# Description :   Introducing histograms + thresholding
# Date:       :   Mar 01, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2

# Import pyplot from matplotlib to visualize
# histograms quickly:
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
fileName = "blueMug.png"

# Reading an image:
inputImage = readImage(path + fileName)

# Get image dimensions
originalImageHeight, originalImageWidth = inputImage.shape[:2]

# Resize at a fixed scale:
resizePercent = 100
resizedWidth = int(originalImageWidth * resizePercent / 100)
resizedHeight = int(originalImageHeight * resizePercent / 100)

# Resize image
inputImage = cv2.resize(inputImage, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Show image:
showImage("grayscaleImage", grayscaleImage)

# Create the histogram
# Returns the count per bin and the bin limits:
histogram, binLimits = np.histogram(grayscaleImage, bins=256)

# Configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")

# X Axis values:
plt.xlim([0.0, 255.0])

# Plot:
# X axis the bin limits,
# Y Axis pixel count:
# Both arrays need the same size, slice the bin Limits to 0-255
plt.plot(binLimits[0:-1], histogram)
plt.show()

# Fixed Threshold, try to separate the foreground object from
# the background:
# Try values 50, 100, 150, 200, 230
binaryThresh = 50
_, binaryImage = cv2.threshold(grayscaleImage, binaryThresh, 255, cv2.THRESH_BINARY)

# Show image:
showImage("binaryImage", binaryImage)

# Let's compute the objects area:

# Invert the image:
binaryImage = 255 - binaryImage

# Show the image:
showImage("binaryImage [Inverted]", binaryImage)
writeImage(path + "binaryMug-nonProcessed", binaryImage)

# Count white pixels:
objectArea = cv2.countNonZero(binaryImage)

# print the value:
print("Object Area (Raw): " + str(objectArea))

# Try to clean the image applying
# Gaussian Blur:
sigma = (5, 5)
binaryImage = cv2.GaussianBlur(binaryImage, sigma, 0)
showImage("binaryImage [Blurred]", binaryImage)

# Compute threshold again:
_, binaryImage = cv2.threshold(binaryImage, binaryThresh, 255, cv2.THRESH_BINARY)

# Show image, no more compression artifacts:
showImage("binaryImage [Revisted]", binaryImage)
writeImage(path + "binaryMug-processed", binaryImage)

# Count white pixels:
objectArea = cv2.countNonZero(binaryImage)

# print the value:
print("Object Area (Processed): " + str(objectArea))

# Automatic thresholding (Otsu)
automaticThreshold, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU)
showImage("binaryImage [Otsu]", binaryImage)
# Print the threshold value:
print("Otsu's threshold is: " + str(automaticThreshold))
