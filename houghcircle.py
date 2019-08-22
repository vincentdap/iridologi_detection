import cv2
import numpy as np


def hough(img):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        height, width = img.shape
        mask = np.zeros((height, width), np.uint8)
        edges = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=3)
        # cv2.imshow('detected ',gray)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1=50, param2=30, minRadius=0, maxRadius=0)
        for i in circles[0, :]:
            i[2] = i[2] + 4
            # Draw on mask
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)

        # Copy that image using that mask
        masked_data = cv2.bitwise_and(img, img, mask=mask)

        # Apply Threshold
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find Contour
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop masked_data
        crop = masked_data[y:y + h, x:x + w]
        cropimg = np.array(crop, dtype=np.uint8)
        return cropimg