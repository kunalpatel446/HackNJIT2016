#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ApplicationServices/ApplicationServices.h>
#include <unistd.h>
using namespace cv;
CGEventRef move;
int main()
{
	std::cout << "Program initiated by user" << std::endl;
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "ERROR: Webcam failed to initialize"
		return -1;
	}
	else
		std::cout << "Webcam initiated" << std::endl;
	namedWindow("Hack NJIT 2016", CV_WINDOW_AUTOSIZE);
	while(true)
	{
		moveCursor(getEyePos());
	}
	return 0;
}
void moveCursor(int x, y)
{
	move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, CGPointMake(200, 200), kCGMouseButtonLeft);
}
std::array<int, 2> getEyePos(std::string LR)
{
	std::array eyePosition;
	return eyePosition;
}