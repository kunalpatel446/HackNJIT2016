#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
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
	
	return 0;
}
void moveCursor(int x, y)
{
	move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, CGPointMake(200, 200), kCGMouseButtonLeft);

}
int *getEyePos()
{
	
}