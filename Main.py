import cv2
import numpy
import sys
import cv2.cv as cv
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)
eyeRect1 = None
eyeRect2 = None
alpha = 0.5
rects = None
leftX = []
leftY = []
leftWidth = []
leftHeight = []
rightX = []
rightY = []
rightWidth = []
rightHeight = []
accuracyCount = 10
def rectCenter(rect):
	x, y, w, h = rect
	return numpy.array((x + 0.5 * w, y + 0.5 * h))
def moveCursor(img):
	# inputs the eye image
	# triggers mouse movement on eye direction
	# from Hough Transforms documentation
	img = cv2.medianBlur(img, 5)
	h, w = img.shape[:2]
	#cv2.imshow("img", img)
	# cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# circles = cv2.HoughCircles(cimg, cv.CV_HOUGH_GRADIENT, 1, 20,param1=50,param2=30,minRadius=0,maxRadius=0)
	# print "circles are ", circles
	# if circles is not None:
	# 	circles = numpy.around(circles)
	# 	circles = numpy.uint16(circles)
	# 	for i in circles[0,:]:
	# 		cv2.circle(img,(i[0],i[1]),2,(0,255,0),3)

while True:
	ok, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	countL = 0
	for (xf, yf, wf, hf) in faces:
		cv2.rectangle(img, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
		roiGray = gray[yf:yf + hf, xf:xf + wf]
		roiColor = img[yf:yf + hf, xf:xf + wf]
		roiL = gray[yf:yf + hf/2, xf:xf + wf/2]
		roiR = gray[yf:yf + hf/2, xf + wf/2:xf + wf]
		eyeL = eyeCascade.detectMultiScale(roiL)
		for (x, y, w, h) in eyeL:
			leftX.append(x)
			leftY.append(y)
			leftWidth.append(w)
			leftHeight.append(h)
		if len(eyeL):
			medX = numpy.array(leftX)
			medX = int(numpy.median(medX))
			medY = numpy.array(leftY)
			medY = int(numpy.median(medY))
			medWidth = numpy.array(leftWidth)
			medWidth = int(numpy.median(medWidth))
			medHeight = numpy.array(leftHeight)
			medHeight = int(numpy.median(medHeight))
			cv2.rectangle(img, (medX + xf,medY + yf), (medX + xf + medWidth, medY + yf + medHeight), (255, 0, 0), 2)
			leftImg = img[(medY + yf):(yf + medY + medHeight), (xf + medX):(xf + medX + medWidth)]
			#cv2.imshow("img", leftImg)
			moveCursor(leftImg)
		eyeR = eyeCascade.detectMultiScale(roiR)
		for (x, y, w, h) in eyeR:
			rightX.append(x)
			rightY.append(y)
			rightWidth.append(w)
			rightHeight.append(h)
		if len(eyeR):
			medX = numpy.array(rightX)
			medX = int(numpy.median(medX))
			medY = numpy.array(rightY)
			medY = int(numpy.median(medY))
			medWidth = numpy.array(rightWidth)
			medWidth = int(numpy.median(medWidth))
			medHeight = numpy.array(rightHeight)
			medHeight = int(numpy.median(medHeight))
			cv2.rectangle(img, (medX + xf + wf/2,medY + yf), (medX + xf + wf/2 + medWidth, medY + yf + medHeight), (255, 0, 0), 2)
			rightImg = img[(medY + yf):(yf + medY + medHeight), (xf + medX + wf/2):(xf + medX + medWidth + wf/2)]
			imgGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)
			M = cv2.moments(imgGray)
			if int(M['m00']) > 10000:
				posX = int(M['m10']/M['m00'])
				posY = int(M['m01']/M['m00'])
				cv2.circle(img, (2 * posX + medX + xf, posY + medY + yf), 2, (0, 0, 255), 3)
			cv2.imshow("img",img)

			# hsv = cv2.cvtColor(rightImg, cv2.COLOR_BGR2HSV)
			
			# mask = cv2.inRange(hsv, lower_range,higher_range)
			# res = cv2.bitwise_and(rightImg, rightImg, mask= mask)
			# cv2.imshow('frame',res)
			#print hsv.size() 
			#print lower_range.size()
			#print higher_range.size()
			#cv2.imshow('mask',mask)
			#cv2.imshow('res',res)
			#k = cv2.waitKey(5) & 0xFF
			#if k == 27: 
			#	break


			#cv2.imshow("img",rightImg)
		#cv2.imshow("img", img)
	if (cv2.waitKey(30) == 27):
		break
cv2.destroyAllWindows()
