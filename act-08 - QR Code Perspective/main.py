# File        :   act-08.py (Activity 8 of 2022 Spring Vision Course)
# Version     :   1.0.1
# Description :   QR Locator/Perspective
# Date:       :   Apr 24, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import webbrowser


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
fileName = "rawCode02.png"

inputImage = readImage(path + fileName)
showImage("Input Image", inputImage)

# To Grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# To HSV:
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

# Split:
(H, S, V) = cv2.split(hsvImage)

# Show channels:
showImage("H", H)
showImage("S", S)
showImage("V", V)

# Gaussian Blur:
sigma = (3, 3)
filteredImage = cv2.GaussianBlur(H, sigma, 0)
showImage("filteredImage [Gaussian Blur]", filteredImage)

# Otsu:
_, binaryImage = cv2.threshold(filteredImage, 0, 255, cv2.THRESH_OTSU)
showImage("binaryImage [Otsu]", binaryImage)

# Get contours:
contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours:
for c in contours:

    # Approximate the contour to a polygon:
    perimeter = cv2.arcLength(c, True)

    # Approximation accuracy:
    approxAccuracy = 0.05 * perimeter

    # Get vertices. Last flag indicates a closed curve:
    vertices = cv2.approxPolyDP(c, approxAccuracy, True)

    # Print the polygon's vertices:
    verticesFound = len(vertices)
    print("Polygon Vertices: " + str(verticesFound))

    # Prepare inPoints structure:
    inPoints = np.zeros((4, 2), dtype="float32")

    # We have the four vertices that made up the
    # contour approximation:
    if verticesFound == 4:

        # Print the vertex structure:
        print(vertices)

        # Format points:
        for p in range(len(vertices)):
            # Get corner points:
            currentPoint = vertices[p][0]

            # Store in inPoints array:
            inPoints[p][0] = currentPoint[0]
            inPoints[p][1] = currentPoint[1]

            # Get x, y:
            x = int(currentPoint[0])
            y = int(currentPoint[1])

            # Draw the corner points:
            cv2.circle(inputImage, (x, y), 5, (255, 0, 0), 5)

            # Draw corner number:
            cv2.putText(inputImage, str(p + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                        thickness=2)

            # Show them corners:
            showImage("Corners", inputImage)

        # Target image dimensions:
        targetWidth = 440
        targetHeight = targetWidth

        # Target Points:
        outPoints = np.array([
            [targetWidth, 0],  # 1
            [0, 0],  # 2
            [0, targetHeight],  # 3
            [targetWidth, targetHeight]],  # 4
            dtype="float32")

        # Compute the perspective transform matrix and then apply it
        H = cv2.getPerspectiveTransform(inPoints, outPoints)
        rectifiedImage = cv2.warpPerspective(grayscaleImage, H, (targetWidth, targetHeight))

        showImage("rectifiedImage", rectifiedImage)

        # Create the QR code object:
        qrCodeDetector = cv2.QRCodeDetector()

        # Decode QR:
        decodedText, bbox, _ = qrCodeDetector.detectAndDecode(rectifiedImage)

        # Check if a code was found:
        if len(decodedText) > 0:

            # Get the code's four corners:
            points = bbox[0]

            # Convert grayscale to BGR:
            currentCode = cv2.cvtColor(rectifiedImage, cv2.COLOR_GRAY2BGR)

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

            # Let's fucking go:
            webbrowser.open(decodedText)

        else:
            print("QR Code not detected!")