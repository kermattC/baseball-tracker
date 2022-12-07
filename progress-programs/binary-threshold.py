"""
I wondered if I can get the binary threshold of a frame to filter out things other than the baseball. Since the baseball is white, binary thresholding was able to filter out a lot. 
But I wasn't sure what to do with it afterwards so I reached a dead end.
"""
import cv2
import numpy as np

frame_with_ball_cap = cv2.VideoCapture(cv2.samples.findFile("../data/pitch.mp4"))
frame_with_ball_cap.set(1, 4)
ret, frame_with_ball = frame_with_ball_cap.read()
frame_with_ball_cap.release()

gray_frame_with_ball = cv2.cvtColor(frame_with_ball, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray_frame_with_ball,(13,13),0)
thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('original', frame_with_ball)
cv2.imshow('output', thresh)
cv2.waitKey(0)
cv2.destroyAllWinsdows()