#pragma once  
#include <opencv2/opencv.hpp>  

using namespace std;  
using namespace cv;  

/**  
* @brief Detects objects in the given frame.  
*  
* If `compute` is set to `true`, the function will call `ComputeObjectCenterAndAngle`  
* to calculate the center and angle of the detected object's largest contour.  
* If `compute` is set to `false`, the function will skip this computation.  
*  
* @param frame The input image frame in which objects are to be detected.  
* @param compute A boolean flag to determine whether to compute the object's center and angle.  
*/  
void ObjectDetection(Mat& frame, bool compute);  

/**  
* @brief Computes the center and angle of the largest contour.  
*  
* This function is called only when `compute` is set to `true` in `ObjectDetection`.  
*  
* @param largestContour A vector of points representing the largest contour of the detected object.  
*/  
void ComputeObjectCenterAndAngle(vector<Point> largestContour);  

/**  
* @brief Calculates the length of the detected object.  
*  
* @return The length of the detected object as a floating-point value.  
*/  
float ObjectLength();
