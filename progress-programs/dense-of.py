# Using dense optical flow via Farneback algorithm. Most of the code comes from the demo from opencv docs: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
import cv2 as cv
import numpy as np
import math                

cap = cv.VideoCapture(cv.samples.findFile("../data/pitch2.mp4"))

# cap = cv.VideoCapture(cv.samples.findFile("data/ran.mp4"))
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

optflow_params = [0.5, 3, 15, 3, 5, 1.2, 0]

prev_flow = []
nearby_flows = [0] * 9
# for i in range(9):
#     nearby_flows[i] = []
track_x = 616
track_y = 323

# colour for bounding box
c = (255, 0, 0)

while(1):
    ret, frame2 = cap.read()
    # print(frame2)
    if not ret:
        print('No frames grabbed!')
        break
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
    
    # (608, 290) pixel with ball
    # (1167, 630) pixel that doesn't move
    # print("Flow on pixel (608, 290) (should move a lot or something): ", flow[track_y][track_x])
    # print("Flow on pixel (1167, 630) (should have 0 movement): ", flow[630][1167])
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    print("Magnitude:", mag[track_x][track_y])
    print("Angle: ", ang[track_y][track_x])

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('Motion', bgr)
    cv.imshow("Flow", draw_flow(next, flow))

    cv.waitKey()

    prvs = next



cv.destroyAllWindows()