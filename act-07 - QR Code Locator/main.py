# File        :   act-07.py (Activity 6 of 2022 Spring Vision Course)
# Version     :   1.1.1
# Description :   QR Locator
# Date:       :   Apr 18, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2


# Reads image via OpenCV:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Set the resources paths:
path = "D://opencvImages//"
fileName = "rawCode01.png"

inputImage = readImage(path + fileName)
# inputImageCopy = inputImage.copy()

showImage("Input Image", inputImage)

# To Gray
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
showImage("grayscaleImage", grayscaleImage)

# Make unblurred copy of grayscale:
grayscaleImageCopy = grayscaleImage.copy()

# Gaussian Blur:
sigma = (3, 3)
grayscaleImage = cv2.GaussianBlur(grayscaleImage, sigma, 0)
showImage("grayscaleImage [Blurred]", grayscaleImage)

# Set Canny thresholds:
cannyThresh1 = 100
cannyThresh2 = 2 * cannyThresh1

cannyEdges = cv2.Canny(grayscaleImage, cannyThresh1, cannyThresh2)
showImage("cannyEdges", cannyEdges)

# (After) Apply morphology:
morphoKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cannyEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, morphoKernel, iterations=1)  # Iterations : 1, 5, 10
showImage("cannyEdges [Filtered]", cannyEdges)

# Find the EXTERNAL contours on the binary image:
# Change the contour mode from RETR_CCOMP to RETR_EXTERNAL
contours, _ = cv2.findContours(cannyEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create BGR of cannyEdges:
cannyEdgesColor = cv2.cvtColor(cannyEdges, cv2.COLOR_GRAY2BGR)

# Store the code bounding rectangles here:
codeRectangles = []

# Contour counter:
contourCounter = 1

# Look for the outer bounding boxes (no children):
for c in contours:

    # Draw contour:
    color = (0, 0, 255)
    cv2.drawContours(cannyEdgesColor, [c], 0, color, 3)
    # showImage("Contours", cannyEdgesColor)

    # Convert the polygon to a bounding rectangle:
    boundRect = cv2.boundingRect(c)

    # Get the bounding rect data:
    rectX = int(boundRect[0])
    rectY = int(boundRect[1])
    rectWidth = int(boundRect[2])
    rectHeight = int(boundRect[3])

    # Draw rectangle:
    # color = (0, 0, 255)
    # cv2.rectangle(inputImage, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)

    # Draw circle label:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    fontScale = 1
    fontThickness = 2
    cv2.putText(inputImage, str(contourCounter), (rectX, rectY), font, fontScale, color, fontThickness)

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight

    # Estimate the aspect ratio:
    rectAspectRatio = rectHeight / rectWidth

    # Print properties:
    print("C: " + str(contourCounter) + " Area: " + str(rectArea) + " Aspect Ratio: " + str(rectAspectRatio))
    # (Before) showImage("Bounding Rectangles", inputImage)

    # Increase counter:
    contourCounter += 1

    # Contour filter:
    minArea = 50000
    minAspectRatio = 0.9

    # Default rectangle color:
    color = (0, 0, 255)  # Red

    # Target rectangles:
    if rectArea > minArea and rectAspectRatio > minAspectRatio:
        # Draw rectangle
        color = (255, 0, 0)  # Blue

        # Store the rectangle:
        codeRectangles.append((rectX, rectY, rectWidth, rectHeight))

    cv2.rectangle(inputImage, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)
    showImage("Bounding Rectangles", inputImage)

# Create the QR code object:
qrCodeDetector = cv2.QRCodeDetector()

# Crop the code:
for i in range(len(codeRectangles)):
    # Get code bounding rect:
    currentRectangle = codeRectangles[i]

    # (After) Include Offset
    offSet = 0

    # New dimensions after offset:
    rectX = int(currentRectangle[0] + offSet)
    rectY = int(currentRectangle[1] + offSet)
    rectWidth = int(currentRectangle[2] - 2 * offSet)
    rectHeight = int(currentRectangle[3] - 2 * offSet)

    # Crop
    grayscaleImage = grayscaleImageCopy
    currentCode = grayscaleImage[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
    showImage("currentCode: " + str(i), currentCode)

    # Decode QR:
    decodedText, bbox, _ = qrCodeDetector.detectAndDecode(currentCode)

    # Check if a code was found:
    if len(decodedText) > 0:

        # Get the code's four corners:
        points = bbox[0]

        # Convert grayscale to BGR:
        currentCode = cv2.cvtColor(currentCode, cv2.COLOR_GRAY2BGR)

        # Draw the corners:
        for j in range(len(points)):

            # Coordinates of the point:
            x = int(points[j][0])
            y = int(points[j][1])

            # Draw point:
            color = (0, 0, 255)
            cv2.line(currentCode, (x, y), (x, y), color, 10)

            # Show Image:
            showImage("Code", currentCode)

        print("Decoded Data : {}".format(decodedText))

    else:
        print("QR Code not detected")
        # cv2.imshow("Results", inputImage)