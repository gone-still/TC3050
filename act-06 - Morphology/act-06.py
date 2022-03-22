# File        :   act-06.py (Activity 6 of 2022 Spring Vision Course)
# Version     :   1.0.0
# Description :   Morphology Demo
# Date:       :   Mar 21, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2


# Read an image:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage, delay=0):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Set image path:
path = "D://opencvImages//"
fileName = "coins.png"

# Read image:
inputImage = readImage(path + fileName)

# Show Image:
showImage("Input Image", inputImage)

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Option 1: Otsu
automaticThreshold, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
showImage("binaryImage [Otsu 1]", binaryImage)

# Apply Morphology:

# Set kernel (structuring element) size:
kernelSize = 5  # 5
# Set operation iterations:
opIterations = 1
# Get the structuring element:
morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
# Check out the kernel:
print(morphKernel)

# Perform Dilate:
dilateImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations,
                               cv2.BORDER_REFLECT101)
# Check out the image:
showImage("Dilation", dilateImage)

# The coins are filled, let's try to delete the small noise with
# an Erosion:

# Set kernel (structuring element) size:
kernelSize = 5  # 5
# Set operation iterations:
opIterations = 2
# Get the structuring element:
morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

# Perform Erosion:
erodeImage = cv2.morphologyEx(dilateImage, cv2.MORPH_ERODE, morphKernel, None, None, opIterations,
                              cv2.BORDER_REFLECT101)
# Check out the image:
showImage("Erosion", erodeImage)

# Let's see the difference between the morphed image and the
# original binary:
imgDifference = binaryImage - erodeImage

# Show the image:
showImage("imgDifference", imgDifference)

# Looks like the filtered image is a little bit (one erosion) smaller
# Let's dilate it one more time to restore the coin's original area:

# Set kernel (structuring element) size:
kernelSize = 5
# Set operation iterations:
opIterations = 1
# Get the structuring element:
morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
# Perform Dilate:
dilateImage = cv2.morphologyEx(erodeImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations,
                               cv2.BORDER_REFLECT101)

# Show the image:
showImage("dilateImage 2", dilateImage)

# Are we there yet?
imgDifference = binaryImage - dilateImage
showImage("imgDifference 2", imgDifference)

# Detect circles via Hough Circles:
circles = cv2.HoughCircles(dilateImage, cv2.HOUGH_GRADIENT, dp=1.0, minDist=90, param1=300, param2=10,
                           minRadius=30, maxRadius=115)

# Circles found:
print("Circles Found: " + str(len(circles[0, :])))

# Integer conversion:
circles = np.round(circles[0, :]).astype("int")

# Set circle counter:
circlesCounter = 1

# Prepare BGR image to draw circles onto:
circleMask = cv2.cvtColor(dilateImage, cv2.COLOR_GRAY2BGR)

# Draw circles:
for (x, y, r) in circles:
    # The function receives the circle's center and its radius:
    cv2.circle(inputImage, (x, y), r, (0, 255, 0), 4)

    # Draw circle label:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # black
    fontScale = 1
    fontThickness = 2
    cv2.putText(inputImage, str(circlesCounter), (x, y), font, fontScale, color, fontThickness)

    # Draw centroid:
    # Setting thickness=-1 fills the circle
    cv2.circle(inputImage, (x, y), 5, (0, 0, 255), thickness=-1)

    # Increment number of circles:
    circlesCounter += 1
    showImage("Detected Circles", inputImage)

# Write Image:
writeImage(path + "detectedCoins", inputImage)
