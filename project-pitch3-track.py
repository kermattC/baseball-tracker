# Using dense optical flow via Farneback algorithm. Most of the code comes from the demo from opencv docs: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

# Results

# pitch2.mp4 - best result. ball loses tracking when crossing with the blue jays logo at the back, since it is white. Because of this, dense optical flow detects it as little to no motion
# pitch3.mp4 - ball is moving too fast, loses tracking shortly after the ball begins breaking (curving away)
# pitch-fastball.mp4 - ball becomes occluded with the other players in the video, losing
import cv2 as cv
import numpy as np
import scipy as sp
import math
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
    return I_, loc


# To first get a pixel to track, the ball needs to be found. This will be done via template matching the image gradients
# frame 4 of pitch.mp4 has a nice shot of the baseball with minimal background interference, prior to image gradients
frame_with_ball_cap = cv.VideoCapture(cv.samples.findFile("data/pitch.mp4"))
frame_with_ball_cap.set(1, 4)
ret, frame_with_ball = frame_with_ball_cap.read()
frame_with_ball_cap.release()

# grayscale and perform image gradient on template
gray_ball_frame = cv.cvtColor(frame_with_ball, cv.COLOR_BGR2GRAY)
img_gradient_ball_frame = get_img_gradient(frame_with_ball, gray_ball_frame)

# pick a template from the image
y, x = 282, 615
half_h, half_w = 10, 10
img_gradient_template = pick_patch(img_gradient_ball_frame, y, x, half_h, half_w)

# get the first frame from the video. This will be the part where the pitcher releases the ball
first_frame_cap = cv.VideoCapture(cv.samples.findFile("data/pitch3.mp4"))
first_frame_cap.set(1,1)
ret, first_frame = first_frame_cap.read()
first_frame_cap.release()

# grayscale and perform image gradient on frame
gray_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
img_gradient_frame = get_img_gradient(first_frame, gray_frame)

R = cv.matchTemplate(img_gradient_frame, img_gradient_template, eval(method))

# The location will be used as the starting pixel to track the ball
result, location = highlight(R, img_gradient_template, first_frame, use_max=False)
cv.imshow("Image gradient frame", img_gradient_frame)
cv.imshow("Result", result)

print("Location: {}".format(location))
cv.waitKey()

cap = cv.VideoCapture(cv.samples.findFile("data/pitch3.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
hsv2 = np.zeros_like(frame1)
hsv2[..., 1] = 255

# visualize the direction of movements for each frame. Courtesy of Nicolai Neilsen on Youtube
# Using the return value from opencv's calcOpticalFlowFarneback, it contains velocity and angle
# Velocity is encoded into value (hsv), so brighter pixels means faster/more movement
# Angle is encoded into hue, so different colours are based on movement
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def track_flow(prev_flow, flow):
    print("prev flow: ", prev_flow)
    print("current flow: ", flow)

def find_strongest_flow(nearby_flows, track_x, track_y):
    # find index where the strongest optical flow occurs
    # index = max(nearby_flows)
    # print("Nearby flows: ", nearby_flows)
    # print("Magnitude of top left pixel: ", nearby_flows[0][0])
    nearby_flows = np.reshape(nearby_flows, (5, 5))

    max_value = np.max(nearby_flows)
    max_index = np.where(nearby_flows == max_value)

    flattened = nearby_flows.flatten()
    middle_index = int((len(flattened)) // 2)
    center_pixel = flattened[middle_index]
    center_index = np.where(nearby_flows == center_pixel)
    
    diff_x = center_index[0] - max_index[0]
    diff_y = center_index[1] - max_index[1]

    disp_x = center_index[0] - diff_x
    disp_y = center_index[0] - diff_y

    print("Displacement x: {}, displacement y: {}".format(disp_x, disp_y))

    return disp_x, disp_y

optflow_params = [0.5, 3, 15, 3, 5, 1.2, 0]

prev_flow = []
nearby_flows = [0] * 9

# Select the detected pixel locations from templatre matching to track
track_x = location[0]
track_x += 2 # adding a bit to the x value since it is the brighter part of the ball
track_y = location[1]


# colour for tracking dot
c = (255, 0, 0)

# (621, 325)
# (620, 320)  

# look at nearby pixels
# depending on what majority of pixels the highest magnitudes lie in, the new pixel to be tracked should go towards that direction

disp_x = 0
disp_y = 0
new_track_x = track_x
new_track_y = track_y
while(1):
    ret, frame2 = cap.read()
    # print(frame2)
    if not ret:
        print('No frames grabbed!')
        break
    cv.imshow("Original", frame2)
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 3rd arg: PyrScale - image scale to build pyramids for each image
    # 4th: WinSize - averaging window size. larger value increases robustness to image noise, making it easier to detect fast motion. However might be morre blurred
    # 5th: Iterations - number of iterations the algorithm does at each pyramid level (default is 10)
    # 6th: PolyN - size of pixel neighbourhood used to find polynomial expansion in each pixel. Larger values = image will be approximated with smoother surfaces, resulting in more robust algorithm and more blurred motion field. Default 5, typically 5 or 7
    # 7th: PolySigma -  Standard deviation of Gaussian used to smooth derivatives used as a basis for the polynomial expansion. For PolyN = 5, can set PolySigma = 1.1. For PolyN = 7, can set PolySigma = 1.5. Default 1.1
    # 8th: Gaussian  - Uses Gaussian WinSize x WinSize filter, instead of box filter of same size for optical flow estimation. Using this gives z more accurate flow than box filter, but slower.
    # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.25, 3, 15, 3, 7, 1.5, 0)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, *optflow_params)
    
    # convert optical flow to magnitude and angle
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    print("Magnitude:", mag[track_x][track_y])
    print("Angle: ", ang[track_y][track_x])

    # used for visualizing optical flow gradient. Angle is encoded into hue and magnitude into value
    hsv[..., 0] = ang * 180/ np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    print("Tracking pixel ({},{})".format(new_track_x, new_track_y))
    # draw dot over tracked pixel
    frame_with_dot = cv.rectangle(frame2, (new_track_x, new_track_y),  (new_track_x + 2, new_track_y + 2), c, 4)
    cv.imshow("Result", frame_with_dot)
    cv.imshow('Motion', bgr)
    cv.imshow("Flow", draw_flow(next, flow))

    # Create a 5x5 signal around the current pixel being tracked
    # This will be used to compute the average of each pixel in the kernel
    # Then perform 2D convolution via 3x3 averaging kernel to find which pixel in the kernel has the largest average
    # The result will be location of the next pixel

    signal = np.zeros((5,5), dtype=float)
    kernel = np.ones((9), dtype=float)

    for row in range(len(signal)):
        for col in range(len(signal[0])):
            # print("Iterating row {} column {}".format(row,col))
            signal[row][col] = mag[track_y - 2 + row][track_x - 2 + col]
        
    max_value = np.max(signal)
    max_index = np.where(signal == max_value)[0][0]
    print("Normal index with highest magnitude: {}, value: {}".format(max_index, max_value))
    signal = signal.flatten()
    # print("Reshaped signal: ", signal)


    #  perform moving average
    full_average = np.convolve(signal, kernel/(len(signal)), mode = 'full')
    same_average = np.convolve(signal, kernel/(len(signal)), mode = 'same')
    valid_average = np.convolve(signal, kernel/(len(signal)), mode = 'valid')

    print("Full average: ", full_average)
    f_avg_max_value = np.max(full_average)
    f_avg_max_index = np.where(full_average == f_avg_max_value)
    print("Full average shape: ", np.shape(full_average))
    print("Full average index with highest magnitude: {}, value: {}".format(f_avg_max_index, f_avg_max_value))

    print("Same average: ", same_average)
    s_avg_max_value = np.max(same_average)
    s_avg_max_index = np.where(same_average == s_avg_max_value)
    print("Same average shape: ", np.shape(same_average))
    print("Same average index with highest magnitude: {}, value: {}".format(s_avg_max_index, s_avg_max_value))
  
    print("Valid average: ", valid_average)
    v_avg_max_value = np.max(valid_average)
    v_avg_max_index = np.where(valid_average == v_avg_max_value)
    print("Valid average shape: ", np.shape(valid_average))
    print("Valid average index with highest magnitude: {}, value: {}".format(v_avg_max_index, v_avg_max_value))

    disp_x, disp_y = find_strongest_flow(same_average, new_track_x, new_track_y)

    # new_track_x = track_x + int(disp_x)
    # new_track_y = track_y + int(disp_y)

    print("track_x: {}, track_y: {}, disp_x: {}, disp_y: {}".format(track_x, track_y, disp_x, disp_y))
    new_track_x += int(disp_x)
    new_track_y += int(disp_y)
 
    cv.waitKey()

    prvs = next

    # disp_x = int(mag[track_y][track_x] / math.sin(ang[track_y][track_x]))
    # disp_y = int(mag[track_y][track_x] / math.sin(ang[track_y][track_x]))

    print("disp_x: {}, disp_y: {}".format(disp_x, disp_y))
    # prev_flow = flow[track_y][track_x]

cv.destroyAllWindows()