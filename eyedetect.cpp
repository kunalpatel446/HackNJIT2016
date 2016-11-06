#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <thread>
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
cv::Rect eyeBb;
std::vector<cv::Rect> faces, eyes;
cv::Rect rect;
void detectAndDisplay()
{
  Mat frame_gray;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  //-- Detect faces
  faceCascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
  for(int i = 0; i < faces.size(); i++)
  {
    Mat face = frame_gray(faces[0]);
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
  }
}
void drawRects()
{
  for(int i = 0; i < eyes.size(); i++)
    rect = eyes[i] + cv::Point(faces[0].x, faces[0].y);
}
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
  bSuccess = cap.read(frame);
  while(true)
  {
    //moveCursor(getEyePos());
    bSuccess = cap.read(frame);
    if (!bSuccess)
    {
      std::cout << "Frame dropped" << std::endl;
      break;
    }
    if(!frame.empty())
    {
      std::thread first (detectAndDisplay);
      std::thread second (drawRects);
      first.join();
      second.join();
    }
    else
    {
      printf(" --(!) No captured frame -- Break!"); break;
    }
    imshow("HackTCNJ2016", frame);
    if (waitKey(30) == 27)
    {
      std::cout << "Program terminated by user" << std::endl;
      break;
    }
  }
  return 0;
}
