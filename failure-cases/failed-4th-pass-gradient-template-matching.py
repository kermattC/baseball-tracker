# I think I messed up some part. This file might be a bust

# 3rd attempt of tracking a baseball
# the previous method of using the image gradient as a domain for template for template matching seems pretty neat, except that it doesn't have scale invariarance
# in this file I'll try implementing scale invariance via gaussian image pyramids, and see if that improves the tracking performance

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
    val = max_val if use_max else min_val

    loc1 = loc + np.array([h//2, w//2])               # Size of R is different from I 
    tl = loc1 - np.array([h//2, w//2])
    br = loc1 + np.array([h//2, w//2])
    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
#     print(c)
#     print(tl)
#     print(br)
    cv.rectangle(I_, tuple(tl), tuple(br), c, 4)
    return I_, loc, val

# Quereshi's code for generating gaussian pyramid
def gen_gaussian_pyramid(I, levels):
    G = I.copy()
    gpI = [G]
    for i in range(levels):
        G = cv.pyrDown(G)
        gpI.append(G)
    return gpI

# Quereshi's code for generating regular image pyramid
def gen_pyramid(I, levels=6):
    G = I.copy()
    pI = [G]
    for i in range(levels):
        G = G[::2,::2,:]
        pI.append(G)
    return pI

# Quereshi's code for generating laplacian image pyramid
def gen_laplacian_pyramid(gpI):
    """gpI is a Gaussian pyramid generated using gen_gaussian_pyramid method found in py file of the same name."""
    num_levels = len(gpI)-1
    lpI = [gpI[num_levels]]
    for i in range(num_levels,0,-1):
        print("level", i)
        GE = cv.pyrUp(gpI[i])
        L = cv.subtract(gpI[i-1],GE)
        lpI.append(L)
    return lpI

def make_square(I):
    h = I.shape[0]
    w = I.shape[1]

    n_levels = int(np.ceil(np.log(np.max([h, w]))/np.log(2)))
    new_h = np.power(2, n_levels)
    new_w = new_h

    if len(I.shape) == 3:
        tmp = np.zeros([new_h, new_w, I.shape[2]], dtype = I.dtype)
        tmp[:h, :w, :] = I
    else:
        tmp = np.zeros([new_h, new_w], dtype = I.dtype)
        tmp[:h, :w] = I

    return tmp, n_levels

def draw_rect(I, bbox):

    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
    cv.rectangle(I_, bbox, c, 4)
    return I_


# frame 55 of k-gausman-pitch2 has a nice shot of the baseball with minimal background interference, prior to image gradients
frame_with_ball_cap.set(1, 4)
ret, frame_with_ball = frame_with_ball_cap.read()
# print(np.shape(frame))
# cv.imshow("Frame with ball", frame_with_ball)

frame_with_ball_cap.release()

gray_ball_frame = cv.cvtColor(frame_with_ball, cv.COLOR_BGR2GRAY)
img_gradient_ball_frame = get_img_gradient(frame_with_ball, gray_ball_frame)

# cv.imshow("Image gradient frame with ball", img_gradient_ball_frame)

# pick a template from the image
y, x = 282, 615
half_h, half_w = 10, 10
regular_template = pick_patch(frame_with_ball, y, x, half_h, half_w)
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



cap = cv.VideoCapture(cv.samples.findFile("../data/pitch.mp4"))
ret, prev_frame = cap.read()
cv.waitKey()

prev_gray_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
img_gradient_frame = get_img_gradient(prev_frame, prev_gray_frame)
temp_gradient_frame = get_img_gradient(img_gradient_ball_frame, gray_ball_frame) 
img_gradient_frame_gray = cv.cvtColor(img_gradient_frame, cv.COLOR_BGR2GRAY)

# trying this number for the levels atm
levels = 3
# squaredImg, levels = make_square(prev_frame)
g_pyr_frame = gen_gaussian_pyramid(temp_gradient_frame, levels)

cv.imshow("Gaussian pyramid level 0", g_pyr_frame[0])
cv.imshow("Gaussian pyramid level 1", g_pyr_frame[1])
cv.imshow("Gaussian pyramid level 2", g_pyr_frame[2])
cv.imshow("Gaussian pyramid level 3", g_pyr_frame[3])

l_pyramid_frame = gen_laplacian_pyramid(g_pyr_frame)
cv.imshow("Laplacian pyramid level 0", l_pyramid_frame[0])
cv.imshow("Laplacian pyramid level 1", l_pyramid_frame[1])
cv.imshow("Laplacian pyramid level 2", l_pyramid_frame[2])
cv.imshow("Laplacian pyramid level 3", l_pyramid_frame[3])

cv.waitKey()

locList = []
valList = []
highlightedImgList = []

counter = 0
for img in g_pyr_frame:
    print("Image size: ", img.size,  "template size: ", img_gradient_template.size)
    if img.size > img_gradient_template.size:
        R = cv.matchTemplate(img, img_gradient_template, eval(method))
        result, loc, val = highlight(R, img_gradient_template, frame_with_ball, use_max=True)
        print("Gaussian pyramid: ", counter, " Location: ", loc, " Val: ", val)
        
        cv.imshow("Image ", img)
        cv.imshow("Result from highlight: ", result)
        cv.waitKey()

        locList.append(loc)
        valList.append(val)
        highlightedImgList.append(result)
    counter += 1

maxVal = min(valList)
maxValIndex = valList.index(maxVal)

w, h = temp_gradient_frame.shape[0], temp_gradient_frame.shape[1]
loc1 = locList[maxValIndex] + np.array([h//2, w//2])
tl = loc1 - np.array([h//2, w//2])
br = loc1 + np.array([h//2, w//2])

box = [(2**maxValIndex)*tl[0], (2**maxValIndex)*tl[1], (2**maxValIndex)*h, (2**maxValIndex)*w]

resultImg = draw_rect(frame_with_ball, box)
cv.imshow("Result", resultImg)
cv.waitKey()
# using gaussian and laplacian pyramid, perform template matching at each scale
# find the correct scale and location within that scale using maxima (minima) search

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