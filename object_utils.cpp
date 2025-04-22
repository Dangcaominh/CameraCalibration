#include <opencv2/opencv.hpp>
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
    if (compute)
    {
		ComputeObjectCenterAndAngle(largestContour);
    }

    // Vẽ kết quả
    Mat image = frame.clone();
    cv::Point2f rect_points[4];
    box.points(rect_points);
    for (int i = 0; i < 4; i++) 
    {
        cv::line(image, rect_points[i], rect_points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Object Detection", image);
}

void ComputeObjectCenterAndAngle(vector<Point> largestContour)
{
    // Tính tâm vật thể
    cv::Moments M = cv::moments(largestContour);
    int cx = int(M.m10 / M.m00);
    int cy = int(M.m01 / M.m00);

    // Tính góc bằng bounding box xoay
    box = cv::minAreaRect(largestContour);
    float angle = box.angle;
    std::cout << "Tam vat: (" << cx << ", " << cy << ")" << std::endl;
    std::cout << "Goc xoay: " << angle << " do" << std::endl;

}

// Hiển thị ảnh
float ObjectLength(Mat& frame)
{
	float length = max(box.size.width, box.size.height);
	return length;
}