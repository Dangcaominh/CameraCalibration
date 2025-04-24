#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp> 
#include "object_utils.h"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;



RotatedRect box;

void ObjectDetection(Mat& frame, bool compute)
{
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    Mat blurred, thresh;
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    threshold(blurred, thresh, 100, 255, THRESH_OTSU);
    Mat edge;
    Canny(thresh, edge, 50, 150, 3);

    // Tìm contour
    vector<vector<Point>> contours;
    findContours(edge, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        cerr << "Object not found!" << std::endl;
        return;
    }

    // Chọn contour lớn nhất (vật thể lớn nhất)
    size_t largestContourIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestContourIdx = i;
        }
    }

    vector<Point> largestContour = contours[largestContourIdx];
    box = minAreaRect(largestContour);
    if (compute)
    {
		ComputeObjectCenterAndAngle(largestContour);
		imwrite("ObjectDetection.jpg", frame);
    }


    // Vẽ kết quả
    Mat image = frame.clone();
    Point2f rect_points[4];
    box.points(rect_points);
    for (int i = 0; i < 4; i++) 
    {
        line(image, rect_points[i], rect_points[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }

    imshow("Object Detection", image);
}

void ComputeObjectCenterAndAngle(vector<Point> largestContour)
{
    // Tính tâm vật thể
    cv::Moments M = cv::moments(largestContour);
    int cx = int(M.m10 / M.m00);
    int cy = int(M.m01 / M.m00);

    
    float angle = box.angle;
    cout << "Tam vat: (" << cx << ", " << cy << ")" << std::endl;
    cout << "Goc xoay: " << angle << " do" << std::endl;

}

extern Mat intrinsicMatrix, rotationMatrix, translationVector;

float ObjectLength()
{
   Point2f rect_points[4];
   box.points(rect_points); // Lấy 4 góc của RotatedRect
   cout << "4 goc: " << rect_points[0] << " " << rect_points[1] << " " << rect_points[2] << " " << rect_points[3] << "\n";
   Mat P = Mat::zeros(3, 4, CV_64F);
   for (int i = 0; i < 3; i++)
   {
	   for (int j = 0; j < 3; j++)
	   {
		   P.at<double>(i, j) = rotationMatrix.at<double>(i, j);
	   }
   }
   P.at<double>(0, 3) = translationVector.at<double>(0, 0);
   P.at<double>(1, 3) = translationVector.at<double>(1, 0);
   P.at<double>(2, 3) = translationVector.at<double>(2, 0);

   P = intrinsicMatrix * P; // Tính ma trận nội suy

   Mat transformedP = Mat::zeros(3, 3, CV_64F);
   for (int i = 0; i < 3; i++)
   {
	   transformedP.at<double>(i, 0) = P.at<double>(i, 0);
	   transformedP.at<double>(i, 1) = P.at<double>(i, 1);
	   transformedP.at<double>(i, 2) = P.at<double>(i, 3);
   }


   // Chuyển đổi rect_points sang tọa độ thực tế
   vector<Point3f> realPoints(4);
   for (int i = 0; i < 4; i++)
   {
       Mat point2D = (Mat_<double>(3, 1) << rect_points[i].x, rect_points[i].y, 1);
       Mat transformedPoint = transformedP.inv() * point2D;
	   realPoints[i] = Point3f(transformedPoint.at<double>(0, 0) / transformedPoint.at<double>(2, 0), transformedPoint.at<double>(1, 0) / transformedPoint.at<double>(2, 0), 0);
   }
   // Tính khoảng cách giữa hai điểm đối diện để xác định chiều dài  
   
    // Tính khoảng cách giữa hai điểm đối diện để xác định chiều dài  
    float length1 = sqrt(pow(realPoints[0].x - realPoints[1].x, 2) + pow(realPoints[0].y - realPoints[1].y, 2));  
    float length2 = sqrt(pow(realPoints[1].x - realPoints[2].x, 2) + pow(realPoints[1].y - realPoints[2].y, 2));
   // Chiều dài thực tế là giá trị lớn hơn giữa hai chiều  

   return max(length1, length2);
}