import cv2
import numpy
import sys
import cv2.cv as cv

from Quartz.CoreGraphics import CGEventCreateMouseEvent
from Quartz.CoreGraphics import CGEventPost
from Quartz.CoreGraphics import kCGEventMouseMoved
from Quartz.CoreGraphics import kCGEventLeftMouseDown
from Quartz.CoreGraphics import kCGEventLeftMouseDown
from Quartz.CoreGraphics import kCGEventLeftMouseUp
from Quartz.CoreGraphics import kCGMouseButtonLeft
from Quartz.CoreGraphics import kCGHIDEventTap


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

def mouseEvent(type, posx, posy):
    	theEvent = CGEventCreateMouseEvent(
                    None, 
                    type, 
                    (posx,posy), 
                    kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, theEvent)

def mousemove(posx,posy):
        mouseEvent(kCGEventMouseMoved, posx,posy);

def mouseclick(posx,posy):
        # uncomment this line if you want to force the mouse 
        # to MOVE to the click location first (I found it was not necessary).
        #mouseEvent(kCGEventMouseMoved, posx,posy);
        mouseEvent(kCGEventLeftMouseDown, posx,posy);
        mouseEvent(kCGEventLeftMouseUp, posx,posy);

def rectCenter(rect):
	x, y, w, h = rect
	return numpy.array((x + 0.5 * w, y + 0.5 * h))
def moveCursor(img, posX1, posY1, posX2, posY2):
	# inputs the eye image
	# triggers mouse movement on eye direction
	# from Hough Transforms documentation
	height, width = img.shape[:2]
	xavg = (posX1 + posX2)/2
	yavg = (posY1 + posY2)/2
	xperc = xavg/width
	yperc = yavg/height
	return xperc, yperc

while True:
	ok, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	countL = 0
	for (xf, yf, wf, hf) in faces:
		#cv2.rectangle(img, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
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
			imgGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
			M = cv2.moments(imgGray)
			if int(M['m00']) > 10000:
				posX1 = int(M['m10']/M['m00'])
				posY1 = int(M['m01']/M['m00'])
				cv2.circle(img, (4 * posX1 + medX + xf, posY1 + medY + yf), 2, (0, 0, 255), 3)
			#cv2.imshow("img",img)

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
				posX2 = int(M['m10']/M['m00'])
				posY2 = int(M['m01']/M['m00'])
				cv2.circle(img, (2 * posX2 + medX + xf, posY2 + medY + yf), 2, (0, 0, 255), 3)
			#cv2.imshow("img",img)`
			xperc, yperc = moveCursor(img, posX1, posY1, posX2, posY2)
			height, width = img.shape[:2]
			xpos = xperc*width
			ypos = yperc*height
			mousemove(xpos,ypos)
		cv2.imshow("img", img)
	if (cv2.waitKey(30) == 27):
		break
cv2.destroyAllWindows()
