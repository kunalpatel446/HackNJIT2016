import cv2
import numpy
import sys
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
pinDownR = []
accuracyCount = 10

def rectCenter(rect):
	x, y, w, h = rect
	return numpy.array((x + 0.5 * w, y + 0.5 * h))

def pupilCenter(img):
	# inputs the median image 
	# no outputs ATM, but draws dot on center. Later outputs coordinates of center
	# from Hough Transforms documentation
	img = cv2.medianBlut(img, 5)
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.20, param1=50, param2=30, minRadius=0, maxRadius=0)
	cicles = numpy.uint16(np.around(circles))
	for i in circles[0,:]:
		cv2.circle(cimg,(i[0],i[2]),(0,255,0),2)

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
			countL +=1
			leftX.append(x)
			leftY.append(y)
			leftWidth.append(w)
			leftHeight.append(h)
			#if (len(pinDownL) == accuracyCount - 1):
				#sumW= []
				#sumH= []
				#for i in range(len(pinDownL[0])):
					#sumW.append(pinDownL[i][0])
					#sumH.append(pinDownL[i][1])
		if eyeL is not None:
			medX = numpy.array(leftX)
			medX = int(numpy.median(medX))
			medY = numpy.array(leftY)
			medY = int(numpy.median(medY))
			medWidth = numpy.array(leftWidth)
			medWidth = int(numpy.median(leftWidth))
			medHeight = numpy.array(leftHeight)
			medHeight = int(numpy.median(leftHeight))
			cv2.rectangle(img, (medX + xf,medY + yf), (medX + xf + medWidth, medY + yf + medHeight), (255, 0, 0), 2)
			#leftImg = img[(medX + xf, medY + yf), (medX + xf + medWidth, medY + yf + medHieght)]
			#pupilCenter(lefImg)

		eyeR = eyeCascade.detectMultiScale(roiR)
		for (x, y, w, h) in eyeR:
			cv2.rectangle(img, (x + xf + wf/2,y + yf), (x + xf + wf/2 + w, y + yf + h), (255, 0, 0), 2)
			if (len(pinDownR) == accuracyCount - 1):
				sumW= []
				sumH= []
				for i in range(len(pinDownL[0])):
					sumW.append(pinDownR[i][0])
					sumH.append(pinDownR[i][1])
				medianWidth = median(sumW)
				medianHieght = median(sumH)	
			#if len(pindownR) == accuracyCount:

		cv2.imshow("img", img)
	if (cv2.waitKey(30) == 27):
		break
cv2.destroyAllWindows()

"""if len(eyeRects):			# Find the eyes and check them
			if eyeRect2 is None:
				if eyeRect1 is None: 	# newly classify when none found before
					x = eyeRects		 #find first eye
					print "eye 1 is", x
					areas = eyeRects[:,2] * eyeRects[:,3]
					print "area1 is", areas
					eyeRect1 = eyeRects[areas.argmin()]
				else:
					x2 = eyeRects 
					print "eye 2 is", x2	
					areas = eyeRects[:,2] * eyeRects[:,3]   # find second eye
					print "area2 is", areas
					eyeRect2 = eyeRects[areas.argmax()]
			else: 					# otherwise chose closest to previous detection
				eyeCenter1 = rectCenter(eyeRect1)
				centers1 = eyeRects[:,:2] + 0.5*eyeRects[:,2:]
				diffs1 = centers1 - eyeCenter1
				p1 = eyeRects[(diffs1*2).sum(axis=1).argmin()]

				eyeCenter2 = rectCenter(eyeRect2)
				centers2 = eyeRects[:,:2] + 0.5*eyeRects[:,2:]
				diffs2 = centers2 - eyeCenter2
				p2 = eyeRects[(diffs2*2).sum(axis=1).argmin()]
				if (numpy.array_equal(p1,p2)): 		# check if eyes the same
					rects = numpy.array([p1])
					eyeRect2 = None
					print "one eye"
				else: 						# if not, do interpolation!
					eyeRect1 = alpha*p1 + (1-alpha)*eyeRect1
					eyeRect2 = alpha*p2 + (1-alpha)*eyeRect2
					rects = numpy.array([eyeRect1,eyeRect2])
					print "two eyes"
			if eyeRect2 is None:
				print "rect2 is none"
			if rects is not None:
				for (ex, ey, ew, eh) in rects:
					cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)"""

