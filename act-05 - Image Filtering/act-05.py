# File        :   act-05.py (Activity 5 of 2022 Spring Vision Course)
# Version     :   1.0.1
# Description :   Filter convolution demo
# Date:       :   Mar 22, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import math

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
fileName = "lena256-color.png"

# Read image:
inputImage = readImage(path + fileName)

# Show Image:
showImage("Input Image", inputImage)

# Get the image dimensions:
imageHeight, imageWidth, imageChannels = inputImage.shape
print((imageHeight, imageWidth, imageChannels))

# Create the output image:
outputImage = np.zeros((imageHeight, imageWidth, imageChannels), np.uint8)

# Set the 3 x 3 kernel:
windowSize = 3

# Set the loop limit for kernel sliding:
loopBound = windowSize - 1

# Convolve the image with the filter:

for j in range(0, imageHeight - loopBound):
    for i in range(0, imageWidth - loopBound):

        # Define the kernel window:
        windowHeight = windowSize
        windowWidth = windowSize

        # Kernel coordinates:
        windowX = i
        windowY = j

        # Numpy Slicing the kernel area
        filterWindow = inputImage[windowY:windowY + windowHeight,
                       windowX:windowX + windowWidth]

        # Show the kernel window:
        showImage("Filter Window (Kernel)", filterWindow)

        # Create a deep copy of the image before modifing it:
        # rectangleImage = inputImage # Shallow copy
        rectangleImage = inputImage.copy() # Deep Copy

        # Draw the kernel area/rectangle on the input image:
        color = (0, 255, 0) # BGR
        cv2.rectangle(rectangleImage, (int(windowX), int(windowY)),(int(windowX + windowWidth),
                       int(windowY + windowHeight)), color, 1)

        # Show images:
        showImage("Sliding Window", rectangleImage)
        showImage("Input Image", inputImage)

        # This variable stores the average of a pixel:
        pixelAverage = [0.0, 0.0, 0.0]

        # Loop through the kernel:
        for y in range(windowSize):
            for x in range(windowSize):

                # Get current BGR pixel:
                currentPixel = filterWindow[y, x]

                # Get pixel channels:
                b = currentPixel[0] # Blue
                g = currentPixel[1] # Green
                r = currentPixel[2] # Red

                # Sum/Acumulate the current channel value to previous one:
                pixelAverage[0] = b + pixelAverage[0]
                pixelAverage[1] = g + pixelAverage[1]
                pixelAverage[2] = r + pixelAverage[2]


        # Compute average of accumulations:
        factor = 1.0/float(windowSize*windowSize)
        pixelAverage[0] = int(factor * pixelAverage[0])
        pixelAverage[1] = int(factor * pixelAverage[1])
        pixelAverage[2] = int(factor * pixelAverage[2])

        # Compute the output coordinates:
        # They must be INTEGER numbers:
        outY = j + math.floor(0.5 * windowSize)
        # outY = j + (windowSize//2)
        outX = i + math.floor(0.5 * windowSize)
        # outX = i + (windowSize//2)

        # Set the pixel average:
        outputImage[outY, outX] = pixelAverage

        # Check out the output image:
        # showImage("Output Window", outputImage)


# Row to row processing:
# showImage("Filter Window (Kernell)", filterWindow)
# showImage("Sliding Window", rectangleImage)
showImage("Output Window", outputImage)

# Low pass filter (vectorized:)
windowSize = 3
# Create the kernel:
smallBlur = np.ones((windowSize, windowSize), dtype="float")
# Set the kernel with averaging coefficients:
smallBlur = (1.0/(windowSize*windowSize)) * smallBlur

print("Low pass kernel: ")
print(smallBlur)

# Apply filter to image:
# -1 infers the data type of the output image from input image:
imageBlur = cv2.filter2D( inputImage, -1, smallBlur )
showImage( "Image Blur", imageBlur )

# High pass from a defined kernel:
# Laplacian Kernel:
highPassKernell = np.array([[0,-1,0],
                            [-1,5,-1],
                            [0,-1,0]])

print("High pass kernel: ")
print(highPassKernell)

imageSharp = cv2.filter2D( inputImage, -1, highPassKernell )
showImage( "imageSharp", imageSharp )