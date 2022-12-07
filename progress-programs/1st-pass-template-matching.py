# First attempt of tracking a baseball, simply by using template matching. Not very accurate
# Obviously wouldn't be this easy, but just wanted to document this process

import cv2 as cv
import numpy as np
import scipy as sp
from scipy import signal

frame_with_ball_cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))

def pick_patch(I, y, x, half_h, half_w):
    return I[y-half_h:y+half_h+1, x-half_w:x+half_w+1, :]

# frame 4 has a nice shot of the baseball with minimal background interference, prior to image gradients
frame_with_ball_cap.set(1, 4)
ret, frame_with_ball = frame_with_ball_cap.read()
# print(np.shape(frame))
cv.imshow("Frame with ball frame: ", frame_with_ball)

# pick a template from the image
y, x = 282, 613
half_h, half_w = 10, 10
template = pick_patch(frame_with_ball, y, x, half_h, half_w)

cv.imshow("Template", template)
cv.waitKey()

def highlight(R, T, I, use_max=True):
    print(np.shape(I))
    W, H = I.shape[0], I.shape[1]
    w, h = T.shape[0], T.shape[1]
    wr, hg = R.shape[0], R.shape[1]
#     print(W,H)
#     print(w,h)
#     print(wr,hg)
        
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(R)
    loc = max_loc if use_max else min_loc
    loc = loc + np.array([h//2, w//2])               # Size of R is different from I 
    tl = loc - np.array([h//2, w//2])
    br = loc + np.array([h//2, w//2])
    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
#     print(c)
#     print(tl)
#     print(br)
    cv.rectangle(I_, tuple(tl), tuple(br), c, 4)
    return I_

cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))
while(1):
    ret, frame = cap.read()

    # break when reach end of video
    if not ret:
        print('No frames grabbed!')
        break

    # flatten template and frame into 1 dimension
    # flatFrame = frame.flatten()
    # template = template.flatten()

    method = 'cv.TM_SQDIFF'

    R = cv.matchTemplate(frame, template, eval(method))
    newImg = highlight(R, template, frame, use_max=False)
    cv.imshow("new image", newImg)

    cv.waitKey()

cap.release()
cv.destroyAllWindows