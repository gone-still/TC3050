import numpy as np
import cv2
from datetime import date, datetime


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
path = "D://opencvImages//sudoku//"
fileName = "dataset.png"

# Dataset info:
writeSamples = False

datasetPath = path + "samples//"
dataSamples = 30
dataClasses = 9

sampleHeight = 70
sampleWidth = sampleHeight

# Data set matrix:
dataSet = np.zeros((dataClasses, dataSamples), np.uint8)
inputImage = readImage(path + fileName)
# inputImageCopy = inputImage.copy()

showImage("Input Image", inputImage)

# To Gray
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
showImage("grayscaleImage", grayscaleImage)

# Otsu:
_, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
showImage("binaryImage", binaryImage)

# Color copy:
binaryColor = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)

# Get Contours:
contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

    # Get contour area:
    blobArea = cv2.contourArea(c)
    print(blobArea)

    minArea = 1000

    if blobArea > minArea:

        # Get Bounding Rect:
        bondingRect = cv2.boundingRect(c)

        # Get the bounding rect data:
        rectX = int(bondingRect[0])
        rectY = int(bondingRect[1])
        rectWidth = int(bondingRect[2])
        rectHeight = int(bondingRect[3])

        color = (0, 255, 0)
        cv2.rectangle(binaryColor, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)
        showImage("binaryColor", binaryColor)

        # Crop Area:
        numbersCrop = binaryImage[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
        showImage("numbersCrop", numbersCrop)

        # Copy:
        numbersCropCopy = numbersCrop.copy()
        numbersCropCopy = cv2.cvtColor(numbersCropCopy, cv2.COLOR_GRAY2BGR)

        # Flood-Fill at corner:
        fillColors = [(255, 255, 255), (0, 0, 0)]
        for i in range(len(fillColors)):
            print(i)
            leftCorner = (0, 0)
            fillColor = fillColors[i]
            cv2.floodFill(numbersCrop, None, leftCorner, fillColor)
            showImage("numbersCrop [Filled]", numbersCrop)

        # St cell dimensions:
        cellHeight = rectHeight // dataClasses
        cellWidth = rectWidth // dataSamples

        print("Cell W: " + str(cellWidth) + " H: " + str(cellHeight))

        dataMatrix = []

        # Loop through de dataset, extract blobs:
        for y in range(dataClasses):

            blobList = []

            for x in range(dataSamples):
                # Set crop:
                blobX = x * cellWidth
                blobY = y * cellHeight
                blobW = blobX + cellWidth
                blobH = blobY + cellHeight

                # Crop:
                currentCrop = numbersCrop[blobY:blobH, blobX:blobW]
                blobList.append(currentCrop)

                showImage("currentCrop", currentCrop)

                # Draw Rectangle:
                color = (0, 255, 0)
                cv2.rectangle(numbersCropCopy, (blobX, blobY), (blobW, blobH), color, 2)
                showImage("numbersCropCopy", numbersCropCopy)
                writeImage(path+"datasetMatrix", numbersCropCopy)

            dataMatrix.append(blobList)

        # The matrix of images:
        dataMatrix = np.asarray(dataMatrix)
        (datasetHeight, datasetWidth) = dataMatrix.shape[:2]

        # Get samples dir:
        # samplesDirs = os.listdir(datasetPath)

        # Save the samples:
        for y in range(datasetHeight):
            for x in range(datasetWidth):
                # Get crop:
                currentCell = dataMatrix[y][x]
                showImage("Current Cell", currentCell)
                writeImage(path + "currentCell", currentCell)

                (ch, cw) = currentCell.shape[:2]
                print((ch, cw))

                # Get Contours:
                cellContours, _ = cv2.findContours(currentCell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for s in cellContours:

                    # Get contour area:
                    blobArea = cv2.contourArea(s)
                    print(blobArea)

                    minArea = 10

                    if blobArea > minArea:
                        # Rect:
                        boundingRect = cv2.boundingRect(s)

                        # Canvas:
                        canvas = np.zeros((sampleHeight, sampleWidth, 3), np.uint8)
                        canvasCopy = canvas.copy()
                        print("Canvas W: " + str(sampleWidth) + ", H: " + str(sampleHeight))

                        # Get the bounding rect data:
                        rectX = int(boundingRect[0])
                        rectY = int(boundingRect[1])
                        rectWidth = int(boundingRect[2])
                        rectHeight = int(boundingRect[3])

                        # Crop:
                        currentSample = currentCell[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
                        currentSample = cv2.cvtColor(currentSample, cv2.COLOR_GRAY2BGR)

                        currentSampleCopy = currentSample.copy()

                        print("currentSample W: " + str(rectWidth) + ", H: " + str(rectHeight))

                        cv2.circle(currentSampleCopy, (int(0.5 * rectWidth), int(0.5 * rectHeight)), 2, (255, 0, 0), 5)
                        showImage("currentSampleCopy", currentSampleCopy)

                        ox = int(0.5 * sampleWidth - 0.5 * rectWidth)
                        oy = int(0.5 * sampleHeight - 0.5 * rectHeight)

                        color = (0, 255, 0)  # Blue
                        cv2.circle(canvasCopy, (ox, oy), 2, color, 3)
                        cv2.circle(canvasCopy, (ox, oy), 2, color, 3)

                        canvas[oy:oy + rectHeight, ox:ox + rectWidth] = currentSample
                        canvasCopy[oy:oy + rectHeight, ox:ox + rectWidth] = currentSample
                        cv2.circle(canvasCopy, (int(0.5 * sampleWidth), int(0.5 * sampleHeight)), 2, (0, 0, 255), 3)

                        # Create out path:
                        writePath = datasetPath + str(y+1) + "//" + "s_" + str(x)
                        print(writePath)

                        if writeSamples:
                            writeImage(writePath, canvas)

                        showImage("canvas", canvas)
                        showImage("canvasCopy", canvasCopy)

                        writeImage(path + "finalCanvas", canvas)
