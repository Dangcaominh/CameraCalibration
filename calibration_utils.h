#pragma once
#ifndef FRAME_PROCESS
#define FRAME_PROCESS

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void FrameProcess(Mat& frame);
void ChessBoardCalibration(Mat& frame);


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
pair<Mat, Mat> DirectCalibration(Mat& fineC, Mat& WorldCor, bool option);
void DirectCameraCalibration(Mat& frame);

#endif
