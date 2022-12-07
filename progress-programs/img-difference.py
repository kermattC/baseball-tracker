# uses gaussian difference to find contours around regions that can be identified as "different"
# I don't remember where I found this code, but similar to struct-similarity.py I couldn't find a way to filter out all the other bounding boxes, so this is another dead end

from scipy.ndimage import gaussian_filter
import cv2 as cv
import numpy as np
import imutils


cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))
ret, prvs = cap.read()

while(1):

    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = frame2

    grayPrvs = cv.cvtColor(prvs, cv.COLOR_BGR2GRAY)
    grayNext = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

    # perform gaussian filtering
    gaussPrvs = gaussian_filter(grayPrvs, sigma=1)
    gaussNext = gaussian_filter(grayNext, sigma=1)

    # gettinm the plain difference will wrap around 0 to 255, which will result in weird things
    # diff = grayA - grayB

    # get absolute difference, since we don't want negatives to wrap around 0 to 255
    # diff = cv.absdiff(grayPrvs, grayNext)
    diff = cv.absdiff(grayPrvs, grayNext)
    gaussDiff = cv.absdiff(gaussPrvs, gaussNext)

    # cv.imshow('Original: ', next)
    cv.imshow('Difference:', diff)
    cv.imshow('Guassian Difference:', gaussDiff)
    cv.waitKey()

    # find contours around regions that can be identified as "different"
    # first set the threshold. I kind of just tweaked around the minimum threshold, so it's subject to change
    # 4th arg: I'll just use the basic threshold function, where if the pixel intensity is greater than the threshold, set it to 255
    # 5th arg: Use OTSU thresholding algorithm, where the threshold can be chosen arbitrary. The algorithm finds the optimal threshold value
    thresh = cv.threshold(diff, 0, 255,
	cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # print("Thershold: ", thresh.shape)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
    # print("Contours: ", cnts)
    cnts = imutils.grab_contours(cnts)


    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv.boundingRect(c)
        # cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(prvs, (x, y), (x + w, y + h), (0, 0, 255), 2)  
    # show the output images
    # cv.imshow("Bounding box without gaussisan smooothing", prvs)

    gaussThresh = cv.threshold(gaussDiff, 0, 255,
	cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # print("Thershold: ", thresh.shape)
    # gaussCnts = cv.findContours(gaussThresh.copy(), cv.RETR_EXTERNAL,
	# cv.CHAIN_APPROX_SIMPLE)

    # find the contours. Contours are curves that join all continous paths that has the same color or intensity
    # 3rd argument is the contour approximation method, which affects how the (x, y) coordinates for the contours are stored. If CHAIN_APPROX_NONE is chosen, all boundary points will be stored.
        # if CHAIN_APPROX_SIMPLE is used, only the two end points are stored. ie: if you found the contour of a straight line. Saves memory

    gaussCnts = cv.findContours(gaussThresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.imshow('test', cv.drawContours(gaussDiff, gaussCnts, 0, (0, 255, 0), 3))
    # cv.imshow('test', gaussCnts)
    # cv.drawContours(gausDiff, gaussCnts, -1, (0, 255, 0), 3)
    # print("Contours: ", gaussCnts)
    gaussCnts = imutils.grab_contours(gaussCnts)

    for c in gaussCnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv.boundingRect(c)
        # cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(prvs, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv.imshow('Bounding box with gaussian smoothing', prvs)

    gaussCnts = gaussCnts[0].reshape(-1, 2)
    for (x, y) in gaussCnts:
        cv.imshow('temp', cv.circle(prvs, (x, y), 1, (255, 0, 0, ), 3))


    prvs = next

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)
