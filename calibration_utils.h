#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
* @brief Detects and processes a chessboard pattern in the given frame for calibration purposes.
* 
* This function identifies a chessboard pattern in the provided frame and extracts its corner points.
* It stores the detected points in a vector and saves the image with the detected corners.
* 
* @param frame A reference to the input image (Mat) in which the chessboard pattern is to be detected.
*              The image should be in grayscale or color format.
* @return True if the chessboard pattern is successfully detected and enough images (default is 15) are taken, otherwise False.
*/
bool ChessBoardCalibration(Mat& frame);

/**
* @brief Performs direct calibration using fine corner points and world coordinates.
* 
* @param fineC Matrix containing fine corner points.
* @param WorldCor Matrix containing world coordinates.
* @param option Calibration option: 
*        - If option = 0, skew = 0.
*        - If option = 1, skew is non-zero.
* @return A pair of Mat objects representing the calibration results.
*/
pair<Mat, Mat> DirectCalibration(Mat& fineC, Mat& WorldCor);

/**
* @brief Performs direct camera calibration
* 
* @param frame A reference to the input image (Mat) used for calibration.
*              The image should contain a detectable calibration pattern.
* @return True if the chessboard pattern is successfully detected and enough images (default is 2) are taken, otherwise False.
*/
bool DirectCameraCalibration(Mat& frame);
