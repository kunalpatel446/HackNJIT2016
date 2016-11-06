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
void moveCursor()
{
	//move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, CGcvPointMake(loc[0], loc[1]), kCGMouseButtonLeft);
}
int detectEye(Mat& im, std::vector<Mat>* tpl, std::vector<cv::Rect>* rect)
{
	std::cout << "Seriously FUCK POINTERS" << std::endl;
	std::vector<cv::Rect> faces, eyes;
	faceCascade.detectMultiScale(im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20, 20));
	if (faces.size())
	{
		Mat face = im(faces[0]);
		eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20, 20));
		if (eyes.size())
			for (int j = 0; j < eyes.size(); j++)
			{
				rect->at(j) = eyes[j] + cv::Point(faces[0].x, faces[0].y);
				tpl->at(j) = im(rect->at(j));
				std::cout << "Fuck pointers" << std::endl;
			}
	}
	return eyes.size();
}
void trackEye(Mat& im, std::vector<Mat>* tpl, std::vector<cv::Rect>* rect)
{
	if (rect->size())
		for (int i = 0; i < tpl->size(); i++)
		{
			cv::Size size(rect->at(i).width * 2, rect->at(i).height * 2);
			cv::Rect window(rect->at(i) + size - cv::Point(size.width / 2, size.height / 2));
			window &= cv::Rect(0, 0, im.cols, im.rows);
			Mat dst(window.width - tpl->at(i).rows + 1, window.height - tpl->at(i).cols + 1, CV_32FC1);
			matchTemplate(im(window), tpl->at(i), dst, CV_TM_SQDIFF_NORMED);
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);
			if (minVal <= 0.2)
			{
				rect->at(i).x = window.x + minLoc.x;
				rect->at(i).y = window.y + minLoc.y;
			}
			else
				rect->at(i).x = rect->at(i).y = rect->at(i).width = rect->at(i).height = 0;
		}
}
int main(int argc, char** argv)
{
	std::cout << "Program initiated by user" << std::endl;
	faceCascade.load("haarcascade_frontalface_default.xml");
	eyeCascade.load("haarcascade_eye.xml");
	VideoCapture cap(0);
	if (faceCascade.empty() || eyeCascade.empty() || !cap.isOpened())
	{
		std::cout << "ERROR: Some shit broke fam..." << std::endl;
		return -1;
	}
	std::cout << "Webcam initiated" << std::endl;
	namedWindow("Hack NJIT 2016", CV_WINDOW_AUTOSIZE);
	Mat frame;
	std::vector<Mat>* eyeTpl = new std::vector<Mat>(2);
	std::vector<cv::Rect>* eyeBb = new std::vector<cv::Rect>(2);
	bool works;
	cap.read(frame);
	while(true)
	{
		works = true;
		bSuccess = cap.read(frame);
		if (!bSuccess)
		{
			std::cout << "Frame dropped" << std::endl;
			break;
		}
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		imshow("HackTCNJ2016", frame);
		for (int i = 0; i < eyeBb->size(); i++)
			if (eyeBb->at(i).width == 0 && eyeBb->at(i).height == 0)
				works = false;
		//if (eyeBb->size() > 0 && eyeBb->size() < 2)
		//{

		// }
		// if (eyeBb->at(0).width == 0 && eyeBb->at(0).height == 0 && eyeBb->at(1).width == 0 && eyeBb->at(1).height == 0)
		// {
			std::cout << "1" << std::endl;
			detectEye(gray, eyeTpl, eyeBb);
			for (int i = 0; i < eyeBb->size(); i++)
				if (eyeBb->at(i).width != 0 || eyeBb->at(i).height != 0)
					works = true;
		//}
		//else
		//{
			std::cout << "2" << std::endl;
			trackEye(gray, eyeTpl, eyeBb);
			for (int i = 0; i < eyeBb->size(); i++)
				rectangle(frame, eyeBb->at(i), CV_RGB(0, 255, 0));
		//}
		if (waitKey(30) == 27)
		{
			std::cout << "Program terminated by user" << std::endl;
			break;
		}
	}
	return 0;
}