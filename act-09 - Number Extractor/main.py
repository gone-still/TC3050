# Imports:
import pytesseract  # tesseract (previous installation)
import numpy as np  # numpy
import cv2  # opencv
import math
import os  # os for paths


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes a png image to disk:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Image path:
fileName = "kxL3a.png"

# Create os-indepent path:
path = "D://opencvImages//"

# Read image:
inputImage = cv2.imread(path + fileName)

# To Grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Binary:
thresh, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
showImage("binaryImage 1", binaryImage)

thresh = 0.7 * thresh
thresh, binaryImage = cv2.threshold(grayscaleImage, thresh, 255, cv2.THRESH_BINARY_INV)
showImage("binaryImage 2", binaryImage)


# Row reduction:
reducedImg = cv2.reduce(binaryImage, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]

print(reducedImg)

w = reducedImg.shape[0]

blobMask = np.zeros((1, w), dtype="uint8")
showImage("blobMask", blobMask)

for i in range(w):
    currentValue = reducedImg[i]
    thresh = 1000
    if currentValue > thresh:
        blobMask[0, i] = 255
    else:
        blobMask[0, i] = 0

showImage("blobMask", blobMask)

colorMask = cv2.cvtColor(blobMask, cv2.COLOR_GRAY2BGR)
colorMaskCopy = colorMask.copy()

contours, hierarchy = cv2.findContours(blobMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

widthList = []
xList = []

for i, c in enumerate(contours):
    # Get the contours bounding rectangle:
    boundRect = cv2.boundingRect(c)

    # Get the dimensions of the bounding rectangle:
    rectX = boundRect[0]
    rectY = boundRect[1]

    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    print((rectX, rectY, rectWidth))

    widthList.append(rectWidth)
    xList.append(rectX)

    # Set bounding rectangle:
    color = (0, 0, 255)

    cv2.line(colorMask, (int(rectX), int(rectY)), (int(rectX), int(rectY)), color, 1)

    showImage("Blobs", colorMask)

meanValue = np.mean(np.array(widthList))
medianValue = np.median(np.array(widthList))

sortedX = np.sort(np.array(xList))

print(meanValue)
print(medianValue)

(imageHeight, imageWidth) = grayscaleImage.shape[:2]

for i, c in enumerate(contours):
    # Get the contours bounding rectangle:
    boundRect = cv2.boundingRect(c)

    # Get the dimensions of the bounding rectangle:
    rectX = boundRect[0]
    rectY = boundRect[1]

    rectWidth = boundRect[2]
    rectHeight = imageHeight

    thresh = 0.5

    if rectWidth > medianValue + 0.5 * medianValue:
        # Set bounding rectangle:
        color = (0, 0, 0)
        rectX = rectX + 0.5 * rectWidth
        cv2.line(blobMask, (int(rectX), int(rectY)), (int(rectX), int(rectY)), color, 1)

    showImage("blobMask", blobMask)

for i in range(imageHeight):
    blobMask = np.concatenate((blobMask, blobMask), axis=0)

showImage("blobMask reshaped", blobMask)
# print(xList)
# print(sortedX)
# lastElement = sortedX[-1]
# firstElement = sortedX[0]
# charactersFound = math.ceil((lastElement - firstElement) / medianValue)
#
# print("charactersFound: " + str(charactersFound))
#
# # medianValue = 21
#
# (imageHeight, imageWidth) = grayscaleImage.shape[:2]
#
# resizedWidth = int(firstElement + charactersFound * medianValue)
# # resizedHeight = int(imageWidth * resizePercent / 100)
#
# showImage("inputImage", inputImage)
#
# resized = cv2.resize(inputImage, (resizedWidth, imageHeight), interpolation=cv2.INTER_LINEAR)
#
# showImage("resized", resized)
#
# firstElement = firstElement * (resizedWidth/imageWidth)
#
# for i in range(charactersFound):
#     rectX = firstElement + i * medianValue
#     rectY = 0
#     rectWidth = medianValue
#     rectHeight = imageHeight
#
#     # Set bounding rectangle:
#     color = (0, 0, 255)
#     cv2.rectangle(inputImage, (int(rectX), int(rectY)),
#                   (int(rectX + rectWidth), int(rectY + rectHeight)), color, 1)
#
#     cv2.rectangle(resized, (int(rectX), int(rectY)),
#                   (int(rectX + rectWidth), int(rectY + rectHeight)), color, 1)
#
#     print((i, rectX))
#
#     showImage("Rects 1", inputImage)
#     showImage("Rects 2", resized)
