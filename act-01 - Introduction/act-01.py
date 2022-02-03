# File        :   act-01.py (Activity 1 of 2022 Spring Vision Course)
# Version     :   1.0.0
# Description :   Learning the ropes with OpenCV + Tesseract
# Date:       :   Jan 24, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# Imports:
import pytesseract  # tesseract (previous installation)
import numpy as np  # numpy
import cv2          # opencv
import os           # os for paths


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
rootDir = "D:"
baseDir = "opencvImages"
subBaseDir = "course"

# Create os-indepent path:
# path = "D://opencvImages//course//" # (Windows)
path = os.path.join(rootDir, baseDir, subBaseDir)

# File name of the image
fileName = "sampleText01.png"

# OpenCV Ver:
print("OpenCV version: "+str(cv2.__version__))

# Reading an image in default mode:
inputImage = cv2.imread(os.path.join(path, fileName))

# Show the image and wait for keyboard input:
cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", inputImage)
cv2.waitKey(0)

# Crop the numbers:
# NumPY Slicing newArray = oldArray[y:y+h, x:x+w]

startY = 860
endY = 1648
startX = 613
endX = 2698

# croppedImage = inputImage
croppedImage = inputImage[startY:endY,startX:endX]

# Show the image and wait for keyboard input:
cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
cv2.imshow("Cropped Image", croppedImage)
cv2.waitKey(0)

# Get image dimensions
originalImageHeight, originalImageWidth = croppedImage.shape[:2]
print("Cropped size is - w: "+str(originalImageWidth)+", h: "+str(originalImageHeight))

# Resize at a fixed scale:
resizePercent = 30
resizedWidth = int(originalImageWidth * resizePercent / 100)
resizedHeight = int(originalImageHeight * resizePercent / 100)

# Resize image
resizedImage = cv2.resize(croppedImage, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)

# Show the image and wait for keyboard input:
cv2.namedWindow("Resized Image", cv2.WINDOW_NORMAL)
cv2.imshow("Resized Image", resizedImage)
cv2.waitKey(0)

# Get new dimensions:
resizedImageHeight, resizedImageWidth = resizedImage.shape[:2]
print("Resized image is - w: "+str(resizedImageWidth)+", h: "+str(resizedImageHeight))

# Convert to grayscale:
grayscaleImage = cv2.cvtColor(resizedImage,cv2.COLOR_BGR2GRAY)

# Show the image and wait for keyboard input:
cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Image", grayscaleImage)
cv2.waitKey(0)

# Convert to black and white (binary)
# Let's manually convert this bad boy into a binary image:
for y in range(resizedImageHeight):

    for x in range(resizedImageWidth):
        # Get current grayscale pixel:
        currentPixel = grayscaleImage[y,x]
        print("Pixel at ("+str(x)+", "+str(y)+") :"+str(currentPixel))
        # Substitute pixel
        if currentPixel < 130:
            # Switch pixel value to black:
            grayscaleImage[y,x] = 0
        else:
            # Switch pixel value to white:
            grayscaleImage[y, x] = 255
        # Check out the results in realtime:
        # cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Binary Image", grayscaleImage)
        # cv2.waitKey(0)

    # Check out the results in realtime:
    # cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
    #cv2.imshow("Binary Image", grayscaleImage)
    #cv2.waitKey(0)

cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
cv2.imshow("Binary Image", grayscaleImage)
cv2.waitKey(0)

# writeImage(path + "firstImage-binary", grayscaleImage)

# Setting up tesseract:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # for Windows
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(grayscaleImage, config=custom_config)

# Show recognized text:
print("Text is: "+text)