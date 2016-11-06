#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <ApplicationServices/ApplicationServices.h>
#include <unistd.h>
using namespace cv;
CGEventRef move;
CascadeClassifier faceCascade, eyeCascade;
bool bSuccess;
Mat frame, eyeTpl;
Rect eyeBb;
int main(int argc, char** argv)
{
	std::cout << "Program initiated by user" << std::endl;
	faceCascade.load("haarcascade_frontalface_default.xml");
	eyeCascade.load("haarcascade_eye.xml");
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		std::cout << "ERROR: Some shit broke fam..." << std::endl;
		return -1;
	}
	else
		std::cout << "Webcam initiated" << std::endl;
	namedWindow("Hack NJIT 2016", CV_WINDOW_AUTOSIZE);
	bSuccess = cap.read(imgTmp);
	while(true)
	{
		//moveCursor(getEyePos());
		bSuccess = cap.read(imgOriginal);
		if (!bSuccess)
		{
			std::cout << "Frame dropped" << std::endl;
			break;
		}
		imshow("HackTCNJ2016", imgOriginal);
		if (waitKey(30) == 27)
		{
			std::cout << "Program terminated by user" << std::endl;
			break;
		}
	}
	return 0;
}
void moveCursor(int *loc)
{
	//move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, CGPointMake(loc[0], loc[1]), kCGMouseButtonLeft);
}
int detectEye(Mat im )