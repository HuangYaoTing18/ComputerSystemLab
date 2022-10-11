import imp
from tracemalloc import is_tracing
from attr import NOTHING
import cv2
from cv2 import putText
from cv2 import waitKey
import numpy as np
# Choose your webcam: 0, 1, ...
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/re_gesture.mp4")

# cv2.namedWindow('Threshold Sliders')
def doNothing(x):
	pass


def recongzize(L):
	multiTouchS = None
	multiTouchT = None
	multiTouch = False
	for i in range(len(L)-1):
		if L[i][0] == L[i+1][0]:
			if multiTouchS is None:
				multiTouchS = (L[i][1], L[i+1][1])
			multiTouchT = (L[i][1], L[i+1][1])
			multiTouch = True
	if multiTouch:
		S1, S2 = multiTouchS
		T1, T2 = multiTouchT
		if S1[0] > S2[0]:
			S1, S2 = S2, S1
		if T1[0] > T2[0]:
			T1, T2 = T2, T1
		if S1[0] < T1[0] and S2[0] > T2[0]:
			return "Zoom in"
		else:
			return "Zoom out"


	else:
		S = L[0][1]
		T = L[-1][1]
		pathLen = np.linalg.norm(np.array(T) - np.array(S))
		print(pathLen)
		for i in range(len(L)-1):
			if L[i][0] + 5 < L[i+1][0]: 				# harded coded frame number
				return "Double Tap"
		eps = 20			# hard coded size
		if pathLen < 1.5*eps:		# hard coded size
			if L[-1][0] - L[0][0] > 20:             	# harded coded frame number
				return "Long Tap"
			return "Tap"
		else:
			hDelta = T[0] - S[0]
			vDelta = T[1] - S[1]
			if abs(hDelta) > abs(vDelta):
				if S[0] - T[0] > eps:
					return "Left"
				else:
					return "Right"
			else:
				if S[1] - T[1] > eps:
					return "Down"
				else:
					return "Up"
		
# cv2.createTrackbar('R', 'Threshold Sliders', 142, 255, doNothing)
# cv2.createTrackbar('delta', 'Threshold Sliders', 100, 200, doNothing)
# cv2.createTrackbar('R-B_thresh', 'Threshold Sliders', 128, 255, doNothing)
#cv2.createTrackbar('R-G+100', 'Threshold Sliders', 100, 200, doNothing)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
blur_kernel = np.ones((3,3),np.float32)/9
frame_count = 0


is_tracing = False
last_time_tracing = -1
img_count = 0
lastX,lastY = None,None

traecdPoints = []

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
	thresR = 30
	thresB = 30
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
	display = zeros

	strokeThickness = 10



	circle_count = 0
	for cnt in contours:
	# Calculate the area of the contour
		area = cv2.contourArea(cnt) 
		# Find the centroid
		(x,y), radius = cv2.minEnclosingCircle(cnt)
		if radius < 30:						# harded coded size
			continue
		circle_count += 1
		traecdPoints.append((frame_count, (int(x), int(y))))
		# print("x: ", x, "y: ", y, "radius: ", radius)
		if is_tracing:
			last_time_tracing = frame_count
		else:
			is_tracing = True
			last_time_tracing = frame_count
	if circle_count == 0 and frame_count - last_time_tracing > 10 and is_tracing: # harded coded frame number
		print(recongzize(traecdPoints))
		is_tracing = False
		lastX,lastY = None,None
		img_count += 1
		traecdPoints = []
		cv2.waitKey(0)
	cv2.imshow("RminusB", RminusB)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()