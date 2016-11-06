import cv2
import numpy
import sys
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)
def rectCenter(rect):
	x, y, w, h = rect
	return np.array((x + 0.5 * w, y + 0.5 * h))
while True:
	ok, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roiGray = gray[y:y + h, x:x + w]
		roiColor = img[y:y + h, x:x + w]
		eyes = eyeCascade.detectMultiScale(roiGray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
		cv2.imshow("img", img)
	if (cv2.waitKey(30) == 27):
		break
cv2.destroyAllWindows()