from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))
# cap = cv.VideoCapture(cv.samples.findFile("data/ran.mp4"))

ret, prvs = cap.read()
print("first frame: ", prvs.shape)
# cv.imshow("first frame: ", prvs)

while(1):
# for i in range(1):
    ret, frame2 = cap.read()
    next = frame2
    if not ret:
        print('No frames grabbed!')
        break
    
    grayA = cv.cvtColor(prvs, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(grayA, grayB, full=True) 
    diff = (diff * 255).astype("uint8")
    # cv.imshow("Gradient", grad)
    cv.imshow('Difference', diff)
    # print("SSIM: {}, difference: {}".format(score, diff))

    # find contours around regions that can be identified as "different"
    thresh = cv.threshold(diff, 0, 255,
	cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    # print("Threshold: ", thresh.shape)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    print("Contours: ", cnts)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv.boundingRect(c)
        # cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(prvs, (x, y), (x + w, y + h), (0, 0, 255), 2)            
    
    # show the output images
    cv.imshow("Frame 2", prvs)
    
    # wait 30 ms between each frame, and get keypress at the end of vid to exit
    cv.waitKey()

    prvs = next


    # cv.imwrite('Frame 1', frame1)
    # cv.imwrite('Frame 2', frame2)
    # cv.imshow("Diff", diff)
    # cv.imshow("Thresh", thresh)
    cv.waitKey(0)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)
    