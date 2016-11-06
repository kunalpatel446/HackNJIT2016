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
Mat computeMatXGradient(const cv::Mat &mat)
{
	Mat out(mat.rows, mat.cols, CV_64F);
	for (int y = 0; y < mat.rows; ++y)
	{
		const uchar* Mr = mat.ptr<uchar>(y);
		double* Or = out.ptr<double>(y);
		Or[0] = Mr[1] - Mr[0];
		for (int i = 0; i < mat.cols - 1; ++i)
			Or[i] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}
	return out;
}