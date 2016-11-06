#include <iostream>
#include <stdio.h>
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

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

 /** Global variables */
 std::string face_cascade_name = "haarcascade_frontalface_default.xml";
 std::string eyes_cascade_name = "haarcascade_eye.xml";
 cv::CascadeClassifier face_cascade;
 cv::CascadeClassifier eyes_cascade;
 std::string window_name = "Capture - Face detection";
 cv::RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   VideoCapture cap(0);
   if(!cap.isOpened())
   		return -1;
   cv::Mat frame;
   cv::Mat frame_gray;
   std::vector<cv::Rect> faces;


   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream
   cap.read(frame);
   frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() ){

       	// display the frame
        //{ detectAndDisplay( frame ); }
       	// greyscale and downsample
  		cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
  		cv::equalizeHist( frame_gray, frame_gray );

  		face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
  		}
       else{ 
       		printf(" --(!) No captured frame -- Break!"); break; 
       	}

       if(waitKey(30) == 27){
       		break;
       	}
   }
   return 0;
 }
}


