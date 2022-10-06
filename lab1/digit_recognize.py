import imp
from tracemalloc import is_tracing
from attr import NOTHING
import cv2
from cv2 import putText
from cv2 import waitKey
import numpy as np
# Choose your webcam: 0, 1, ...
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("numbers.mp4")

# cv2.namedWindow('Threshold Sliders')
def doNothing(x):
	pass
# cv2.createTrackbar('R', 'Threshold Sliders', 142, 255, doNothing)
# cv2.createTrackbar('delta', 'Threshold Sliders', 100, 200, doNothing)
# cv2.createTrackbar('R-B_thresh', 'Threshold Sliders', 128, 255, doNothing)
#cv2.createTrackbar('R-G+100', 'Threshold Sliders', 100, 200, doNothing)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
blur_kernel = np.ones((3,3),np.float32)/9
frame_count = 0


is_tracing = False
last_time_tracing = -1

while(True):
	frame_count += 1
# for i in range(1):
	# Get one frame from the camera
	ret, frame = cap.read()

	height, width, layers = frame.shape
	frame = cv2.resize(frame, (int(width/2), int(height/2)))


	b,g,r = cv2.split(frame)
	zeros = np.zeros(frame.shape[:2], dtype="uint8")

	# thresR = cv2.getTrackbarPos('R', 'Threshold Sliders')
	thresR = 10
	thresB = 10
	# thresB = thresR - cv2.getTrackbarPos('delta', 'Threshold Sliders')+100
	# thresRmB = cv2.getTrackbarPos('R-B_thresh', 'Threshold Sliders')+100
	# thresG = thresR - cv2.getTrackbarPos('R-G+100', 'Threshold Sliders')+100


	_, r = cv2.threshold(r, thresR, 255, cv2.THRESH_BINARY)
	# _, g = cv2.threshold(g, thresG, 255, cv2.THRESH_BINARY)
	_, b = cv2.threshold(b, thresB, 255, cv2.THRESH_BINARY)

	# cv2.imshow("Red", cv2.merge([zeros, zeros, r]))
	# cv2.imshow("Green", cv2.merge([zeros, g, zeros]))
	# cv2.imshow("Blue", cv2.merge([b, zeros, zeros]))
	
	RminusB = cv2.bitwise_and(r, cv2.bitwise_not(b), mask=None)
	RminusB = cv2.GaussianBlur(RminusB, (5, 5), 0)
	RminusB = cv2.threshold(RminusB, 10, 255, cv2.THRESH_BINARY)[1]
	# cv2.imshow("R-B", cv2.merge([RminusB, RminusB, RminusB]))

	contours, hierarchy = cv2.findContours(RminusB,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Draw the contours (for debugging)
	display = cv2.cvtColor(RminusB, cv2.COLOR_GRAY2BGR) 
	cv2.drawContours(display, contours, -1, (0,0,255))
	circle_count = 0
	for cnt in contours:
	# Calculate the area of the contour
		area = cv2.contourArea(cnt) 
		# Find the centroid
		(x,y), radius = cv2.minEnclosingCircle(cnt)
		if radius < 15:
			cv2.circle(display, (int(x), int(y)), int(radius), (0,0,0), -1)
			continue
		circle_count += 1
		if circle_count > 1:
			print("Noise")
			break
		cv2.circle(display, (int(x), int(y)), int(radius), (0, 0, 0), -1)
		cv2.circle(display, (int(x), int(y)), int(radius/3), (255,255,255), -1)
		cv2.circle(display, (int(x), int(y)), int(radius), (255,255,255), -1)
		print("x: ", x, "y: ", y, "radius: ", radius)
		if is_tracing:
			digit_images = cv2.bitwise_or(digit_images, RminusB)
			last_time_tracing = frame_count
		else:
			digit_images = np.zeros(frame.shape[:2], dtype="uint8")
			digit_images = cv2.bitwise_or(digit_images, RminusB)
			is_tracing = True
			last_time_tracing = frame_count
	if circle_count == 0 and frame_count - last_time_tracing > 15 and is_tracing:
		is_tracing = False
		digit_images = cv2.flip(digit_images, 1)
		cv2.imsave("digit_images.png", digit_images)
		cv2.imshow("digit_images", digit_images)
		# print(area,(x,y), radius)
		# cv2.putText(display, str((x,y)), (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 10, color=(0,255,0))
	cv2.imshow("display", display)
	# Split RGB channels


	# Perform thresholding to each channel


	# Get the final result using bitwise operation


	# Find and draw contours


	# Iterate through each contour, check the area and find the center


	# Show the frame
	# cv2.imshow('frame', frame)
	# Press q to quit
	# cv2.waitKey(0)
	if cv2.waitKey(50) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()