# 2nd attempt of tracking a baseball 
# I tried running an image gradient for each frame and noticed something interesting - the ball is very unique - circular, with a unique colour pattern.
# So I have a theory - using the image gradient of each frame and image gradient of the template, I'll attempt to perform template matching in the image gradient domain, rather than a RGB domain. 
# I think that the unique image gradient template might improve the performance
# Seems to work pretty well  except template matching fails during times such as shortly when the ball leaves the pitcher's hand, and when the ball intersects with the strike zone boundary
    # pitch.mp4: Unable to track shortly after the ball is released from the pitcher's hand, and when the ball intersects with the strike zone overlay. But template matching works otherwise
    # pitch2.mp4: Only loses tracking when ball falls below the strike zone
    # pitch3.mp4: Loses tracking when ball falls below strike zone
# However it doesn't work well for hard-pitch.mp4, hard-pitch2.mp4 and hard-pitch3.mp4
    # This happens mainly due to the ball being a different scale, and many other objects in the frame that looks like the ball
    # A proper implementation of scale invariance might be able to fix this

import cv2 as cv
import numpy as np
import scipy as sp
from scipy import signal

# 3x3 sobel filters for edge detection
sobel_x = np.array([[ -1, 0, 1], 
                    [ -2, 0, 2], 
                    [ -1, 0, 1]])
sobel_y = np.array([[ -1, -2, -1], 
                    [  0,  0,  0], 
                    [  1,  2,  1]])

method = 'cv.TM_SQDIFF'
method2 = 'cv.TM_SQDIFF_NORMED'
# method = 'cv.TM_CCORR'

frame_with_ball_cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))

# Quereshi's code for getting an image patch. I'm too lazy to write my own but hey why stress myself out
def pick_patch(I, y, x, half_h, half_w):
    return I[y-half_h:y+half_h+1, x-half_w:x+half_w+1, :]

# basically Quereshi's code for getting the x and y image gradients
def get_img_gradient(frame, gray):
    mask = np.zeros_like(frame)
    mask[..., 1] = 255 # set image saturation to maximum

    filtered_x = cv.filter2D(gray, cv.CV_32F, sobel_x)
    filtered_y = cv.filter2D(gray, cv.CV_32F, sobel_y)

    mag = cv.magnitude(filtered_x, filtered_y)
    orien = cv.phase(filtered_x, filtered_y, angleInDegrees=True)
    orien = orien / 2. # Go from 0:360 to 0:180 
    mask[..., 0] = orien # H (in OpenCV between 0:180)
    mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # V 0:255
    
    bgr = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    return bgr

# Quereshi's code for putting a bounding box using the score of whichever method for template matching
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

# frame 4 of pitch.mp4 has a nice shot of the baseball with minimal background interference, prior to image gradients
frame_with_ball_cap.set(1, 4)
ret, frame_with_ball = frame_with_ball_cap.read()
frame_with_ball_cap.release()

gray_ball_frame = cv.cvtColor(frame_with_ball, cv.COLOR_BGR2GRAY)
img_gradient_ball_frame = get_img_gradient(frame_with_ball, gray_ball_frame)

# cv.imshow("Image gradient frame with ball", img_gradient_ball_frame)

# pick a template from the image
y, x = 282, 615
half_h, half_w = 10, 10
img_gradient_template = pick_patch(img_gradient_ball_frame, y, x, half_h, half_w)

cv.imshow("Image Gradient Template", img_gradient_template)
cv.waitKey()

# # create image mask. Start with dimensions
# # Mask will represent the optical flow in HSV, which has 3 columns in OpenCV
#     # first column = hue range (0 to 179), will be used to represent direction
#     # second column = saturation (0 to 255), will just set to 255 for maximum saturation
#     # third column = value range (0 to 255), will be used to represent magnitude 
# mask = np.zeros_like(first_frame)
# mask[..., 1] = 255 # set image saturation to maximum
# blurred_mask = np.zeros_like(first_frame)
# blurred_mask[..., 1] = 255


# cap = cv.VideoCapture(cv.samples.findFile("../data/pitch-fastball.mp4"))
cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))

ret, prev_frame = cap.read()

prev_gray_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
img_gradient_frame = get_img_gradient(prev_frame, prev_gray_frame)
while(1):
    ret, frame = cap.read()

    # break when reach end of video
    if not ret:
        print('No frames grabbed!')
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_gradient_frame = get_img_gradient(frame, gray_frame)

    R = cv.matchTemplate(img_gradient_frame, img_gradient_template, eval(method))
    result = highlight(R, img_gradient_template, frame, use_max=False)
    cv.imshow("Image gradient frame", img_gradient_frame)
    cv.imshow("Result", result)

    cv.waitKey()

    prev_frame = frame

cap.release()
cv.destroyAllWindows