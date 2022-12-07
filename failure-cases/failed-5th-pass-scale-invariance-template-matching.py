# applying scale invariance template matching
# going to try frame-by-frame TM with canny edge detection

# Seems to perform better and worse than the 3rd pass where it simply used template matching of image derivatives
    # pitch.mp4: Worse overall, tracks for a few frames when the ball is within the strike zone, but loses tracking everywhere else
    # pitch2.mp4: Can't even track the ball
    # pitch3.mp4: Manages to track the ball shortly after the pitcher releases the ball. But loses tracking shortly after
    # hard-pitch.mp4: Can't track the ball
    # hard-pitch2.mp4: Can't track the ball
    # hard-pitch3.mp4: Tracks the ball for a frame or two while it's in the strike zone
# Majority of code is courtesy of Adrian Rosebrock from pyimagesearch.com
    # Source: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv

import numpy as np
import cv2 as cv
import imutils

# Load the template
template = cv.imread('../data/ball.png')
template = cv.cvtColor(template, cv.COLOR_BGR2RGB)
template = cv.Canny(template, 50, 200) # Try canny edge detection. First and second argument are lower/upper thresholds for hysterisis thresholding
(tH, tW) = template.shape[:2]
cv.imshow("Template", template)

cv.waitKey()

cap = cv.VideoCapture(cv.samples.findFile("../data/hard-pitch3.mp4"))

while(1):
    ret, frame = cap.read()

    print("Frame size: ", np.shape(frame))
    # break when reach end of video
    if not ret:
        print('No frames grabbed!')
        break
    frame = cv.resize(frame, (1920, 1080))
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    found = None
    # for each frame, scale it down and perform template matching
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize image according to the scale
        resized = imutils.resize(gray_frame, width = int(frame.shape[1] * scale))
        # record the ratio of the resizing
        r = gray_frame.shape[1] / float(resized.shape[1])

        # break out of the loop if the image size is smaller than the template
        if (resized.shape[0] < tH or resized.shape[1] < tW):
            break
        
        # find edges of resized image, perform template matching via cross correlation
        Iedged = cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(Iedged, template, cv.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        print("Correlation value: ", maxVal, " Correlation location: ", maxLoc)

        # preview the bounding box on the edge detected frame
        # draw a bounding box around the detected region
        clone = np.dstack([Iedged, Iedged, Iedged])
        cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
            (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        clone = cv.resize(clone, (1080, 720))
        cv.imshow("Visualize", clone)
        cv.waitKey()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            print("Found maximum correlation value: ", found)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    # (startX, startY) = (int( (maxLoc[0] * r)), int((maxLoc[1] * r)))
    # (endX, endY) = (int(((maxLoc[0] + tW) * r)), int(((maxLoc[1] + tH) * r )))

    (startX, startY) = (int( (maxLoc[0] * r) / 1.777), int((maxLoc[1] * r) / 1.5))
    (endX, endY) = (int(((maxLoc[0] + tW) * r) / 1.777), int(((maxLoc[1] + tH) * r ) / 1.5))

    # draw a bounding box around the detected result and display the image
    # frame = cv.resize(frame, (1080, 720))
    frame = cv.resize(frame, (1080, 720))
    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv.imshow("Result", frame)
    cv.waitKey()