# File        :   imageUtils.py (Image misc functions)
# Version     :   1.0.0
# Description :   Some helper functions for the classifier example
# Date:       :   Feb 08, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import random


# Function that reads an image:
def readImage(imagePath, colorCode=cv2.IMREAD_COLOR):
    # Open image:
    print("Reading image from: " + imagePath)
    inputImage = cv2.imread(imagePath, colorCode)
    # Check if the operation was sucessfull:
    if inputImage is None:
        print("Could not load Input image.")

    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage, fixedWindow=True, delay=0):
    if fixedWindow:
        cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Clamps a value to a maximum value:
def clampValue(inputValue, limit):
    if inputValue > limit:
        return limit
    else:
        return inputValue


# Creates random shapes in a image of size (h, w):
# Shapes options are: "circle", "square" and "rectangle":

def createShape(shapeString, size):
    # Roll the PRNG:
    random.seed()
    # to lower case
    shapeString = shapeString.lower()
    # create black canvas:
    (h, w) = size
    outImage = np.zeros((h, w, 3), dtype="uint8")
    # Set random color:
    color = tuple(np.random.random(size=3) * 256)
    # Set thickness (filled):
    thickness = -1

    # Create shape:
    if shapeString == "circle":

        # Possible value of "radius":
        lowLimit = int(0.05 * w)
        highLimit = int(0.3 * w)
        radius = random.randint(lowLimit, highLimit)

        # Possible value of "center"
        low = radius
        high = w - (radius)
        x = random.randint(low, high)
        low = radius
        high = h - (radius)
        y = random.randint(low, high)

        # Draw the random circle:
        outImage = cv2.circle(outImage, (x, y), radius, color, thickness)

    elif shapeString == "rectangle":

        # Possible value of "top left":
        low = 0
        high = w - 0.3 * w
        x1 = random.randint(low, high)
        high = h - 0.3 * h
        y1 = random.randint(low + 5, high)
        p1 = (x1, y1)

        # Possible value of "bottom right":
        w1 = random.randint(10, int(0.8 * w))
        h1 = random.randint(10, int(0.8 * h))
        p2 = (clampValue(x1 + w1, w - 5), clampValue(y1 + h1, h - 5))

        # Draw the random rectangle:
        outImage = cv2.rectangle(outImage, p1, p2, color, thickness)

    elif shapeString == "square":

        # Possible value of "top left":
        low = 0
        high = w - 0.3 * w
        x1 = random.randint(low, high)
        high = h - 0.3 * h
        y1 = random.randint(low, high)
        p1 = (x1, y1)
        w1 = random.randint(10, int(0.3 * w))

        # Check out of canvas on x:
        x2 = x1 + w1
        if x2 > w:
            dif = x2 - x1
            x1 = x1 - dif

        # Check out of canvas on y:
        y2 = y1 + w1
        if y2 > h:
            dif = y2 - y1
            y1 = y1 - dif

        p2 = (x2, y2)

        # Draw the random square:
        outImage = cv2.rectangle(outImage, p1, p2, color, thickness)
    else:
        print("Requested shape not found!")

    return outImage
