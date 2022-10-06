import imp
from attr import NOTHING
import cv2
from cv2 import putText
import numpy as np
# Choose your webcam: 0, 1, ...
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("2022-04-12 10-19-54.mp4")

cv2.namedWindow('Threshold Sliders')
def doNothing(x):
	pass
cv2.createTrackbar('R', 'Threshold Sliders', 142, 255, doNothing)
cv2.createTrackbar('delta', 'Threshold Sliders', 100, 200, doNothing)
cv2.createTrackbar('R-B_thresh', 'Threshold Sliders', 128, 255, doNothing)
#cv2.createTrackbar('R-G+100', 'Threshold Sliders', 100, 200, doNothing)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
blur_kernel = np.ones((3,3),np.float32)/9
while(True):
# for i in range(1):
	# Get one frame from the camera
	ret, frame = cap.read()

	height, width, layers = frame.shape
	frame = cv2.resize(frame, (int(width/2), int(height/2)))


	b,g,r = cv2.split(frame)
	zeros = np.zeros(frame.shape[:2], dtype="uint8")

	thresR = cv2.getTrackbarPos('R', 'Threshold Sliders')
	thresB = thresR - cv2.getTrackbarPos('delta', 'Threshold Sliders')+100
	thresRmB = cv2.getTrackbarPos('R-B_thresh', 'Threshold Sliders')+100
	# thresG = thresR - cv2.getTrackbarPos('R-G+100', 'Threshold Sliders')+100


	_, r = cv2.threshold(r, thresR, 255, cv2.THRESH_BINARY)
	# _, g = cv2.threshold(g, thresG, 255, cv2.THRESH_BINARY)
	_, b = cv2.threshold(b, thresB, 255, cv2.THRESH_BINARY)

	cv2.imshow("Red", cv2.merge([zeros, zeros, r]))
	# cv2.imshow("Green", cv2.merge([zeros, g, zeros]))
	cv2.imshow("Blue", cv2.merge([b, zeros, zeros]))
	
	RminusB = cv2.bitwise_and(r, cv2.bitwise_not(b), mask=None)
	# RminusB = cv2.GaussianBlur(RminusB, (5, 5), 0)
	# RminusB = cv2.threshold(RminusB, thresRmB, 255, cv2.THRESH_BINARY)[1]
	cv2.imshow("R-B", cv2.merge([RminusB, RminusB, RminusB]))

	contours, hierarchy = cv2.findContours(RminusB,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Draw the contours (for debugging)
	display = cv2.cvtColor(RminusB, cv2.COLOR_GRAY2BGR) 
	cv2.drawContours(display, contours, -1, (0,0,255))
	for cnt in contours:
	# Calculate the area of the contour
		area = cv2.contourArea(cnt) 
		# Find the centroid
		(x,y), radius = cv2.minEnclosingCircle(cnt)
		# print("x: ", x, "y: ", y, "radius: ", radius)
		# print(area,(x,y), radius)
		# cv2.putText(display, str((x,y)), (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 10, color=(0,255,0))
	cv2.imshow("display", display)	
	# Split RGB channels


	# Perform thresholding to each channel


	# Get the final result using bitwise operation


	# Find and draw contours


	# Iterate through each contour, check the area and find the center


	# Show the frame
	cv2.imshow('frame', frame)
	# Press q to quit
	# cv2.waitKey(0)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()