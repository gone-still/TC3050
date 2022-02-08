# File        :   act-04.py (Activity 4 of 2022 Spring Vision Course)
# Version     :   1.0.0
# Description :   HSV-based Segmentation + Hough's Circle Detector
# Date:       :   Feb 07, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2


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


# Set image path
path = "D://opencvImages//"
fileName = "blueCircles.png"

# Reading an image in default mode:
inputImage = cv2.imread(path + fileName)
showImage("inputImage", inputImage)

# Get input image dimensions:
(imageHeight, imageWidth, imageChannels) = inputImage.shape

print("Image Height: " + str(imageHeight) + " Image Width: " + str(imageWidth) + " Image Channels: " + str(imageChannels))

# Set the low threshold for the 3 channels:
# lowThreshold = [130, 130, 20]
# highThreshold = [255, 255, 110]

lowThreshold = [210, 210, 20]  # BGR
highThreshold = [255, 255, 70]

# Create new image:
circlesMask = np.zeros((imageHeight, imageWidth), dtype="uint8")

# A black image:
showImage("Circles Mask", circlesMask)

# Let's manually create the circle mask based on BGR values:
for y in range(imageHeight):
    for x in range(imageWidth):
        # get BGR pixel:
        (B, G, R) = inputImage[y, x]

        # Check if the pixels is in threshold:
        if B >= lowThreshold[0] and B <= highThreshold[0]:
            if G >= lowThreshold[1] and G <= highThreshold[1]:
                if R >= lowThreshold[2] and R <= highThreshold[2]:
                    circlesMask[y, x] = 255

# Check the image:
showImage("Circles Mask [BGR]", circlesMask)

# Conversion to HSV:
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

# Check the image:
showImage("Input Image [HSV]", hsvImage)

# Set the HSV range:
# We define "Value" first, for blue the range is [85, 100]
# Next, define the "Saturation" level
# We don't want to include white [0-100], let's set
# the range to [100, 255]
# Finally, "Value". We don't want black [0-50], we set
# the range to [50, 255]

# Blue range:
lowThreshold = [85, 100, 50]
highThreshold = [100, 255, 255]

# Gray:
# lowThreshold = [0, 0, 0]
# highThreshold = [255, 100, 150]

# Create new image:
circlesMaskHSV = np.zeros((imageHeight, imageWidth), dtype="uint8")

# Let's manually create the circle mask based on HSV values:
for y in range(imageHeight):
    for x in range(imageWidth):
        # get BGR pixel:
        (H, S, V) = hsvImage[y, x]

        # Check if the pixels is in threshold:
        if H >= lowThreshold[0] and H <= highThreshold[0]:
            if S >= lowThreshold[1] and S <= highThreshold[1]:
                if V >= lowThreshold[2] and V <= highThreshold[2]:
                    circlesMaskHSV[y, x] = 255

# Check the image:
showImage("Circle Mask [HSV]", circlesMaskHSV)

# Using cv2.inRange:
# OpenCV includes a vectorized version of the nested-loops via
# cv2.inRange(lowThresh, highThresh):
# The function accepts the thresholds as NumPY arrays:

lowThreshold = np.array([85, 100, 50])
highThreshold = np.array([100, 255, 255])

# Apply the ranges to HSV image:
hsvMask = cv2.inRange(hsvImage, lowThreshold, highThreshold)

# Show the image:
showImage("Circle Mask [HSV - inRange]", hsvMask)

# Detect the circles via Hough Circles:
# Hough Circle detection is a method for finding circles in an image
# It is notorious for being hard to configure and very sensitive to
# image changes.
# For now, let's use it as a black box.
# We need to parameters to tune out the detector:
# dp - Inverse ratio of the accumulator resolution to the image resolution
# minDist - Minimum distance between he centers of the detected circles
# There are more parameters, but for now this will do.

circles = cv2.HoughCircles(hsvMask, cv2.HOUGH_GRADIENT, 5, 20)
print("Circles Found: " + str(len(circles[0, :])))

# Integer conversion:
# OpenCV has an habit of returning vectors and scalars in a whole vector.
# This is due to the C++ data type underneath.
# Circles is an array of just one element: the actual array of circles,
# To draw the, we need to cast them to integers first:
circles = np.round(circles[0, :]).astype("int")

circlesCounter = 1

# Draw circles:
for (x, y, r) in circles:
    # The function receives the circle's center and its radius:
    cv2.circle(inputImage, (x, y), r, (0, 255, 0), 4)
    # Draw centroid:
    # Setting thickness=-1 fills the circle
    cv2.circle(inputImage, (x, y), 5, (0, 0, 255), thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # black
    fontScale = 1
    fontThickness = 2
    cv2.putText(inputImage, str(circlesCounter), (x, y), font, fontScale, color, fontThickness)

    # Increment number of circles:
    circlesCounter += 1

    showImage("Detected Circles", inputImage)
