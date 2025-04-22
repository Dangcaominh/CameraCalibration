#pragma once
#include <opencv2/opencv.hpp>
#include "calibration_utils.h"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;


// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 10,7 };

// Creating vector to store vectors of 3D points for each checkerboard image
std::vector<std::vector<cv::Point3f> > objpoints;

// Creating vector to store vectors of 2D points for each checkerboard image
std::vector<std::vector<cv::Point2f> > imgpoints;


void ChessBoardCalibration(Mat& frame)
{
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(float(j * 2.5), float(i * 2.5), 0));
    }

    cv::Mat gray;
    // vector to store the pixel coordinates of detected checker board corners 
    std::vector<cv::Point2f> corner_pts;
    bool success;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);


    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
    */
	Mat frame_copy = frame.clone();
    if (success)
    {
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);


        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame_copy, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
        cv::imshow("Image", frame_copy);

		std::string name = "image" + std::to_string(objpoints.size()) + ".jpg";
		cv::imwrite(name, frame);
    }   
    else
    {
		std::cout << "Khong tim thay hinh vuong!" << std::endl;
        return;
    }
    if (objpoints.size() >= 10)
    {
        cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Mat> rvecs, tvecs;

        /*
        Performing camera calibration by
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the
        detected corners (imgpoints)
        */
        std::streambuf* coutBuf = std::cout.rdbuf();
        cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, rvecs, tvecs);
        std::cout << "Output written to txt file" << std::endl;
        std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
        std::cout << "distCoeffs : " << distCoeffs << std::endl;
        for (size_t i = 0; i < rvecs.size(); ++i) {
            std::cout << "Image #" << i + 1 << std::endl;

            // Convert rotation vector to rotation matrix
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);

            std::cout << "Rotation Matrix:\n" << R << std::endl;
            std::cout << "Translation Vector:\n" << tvecs[i] << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }

    }
}

pair<Mat, Mat> DirectCalibration(Mat& fineC, Mat& WorldCor, bool option)
{
    int n = fineC.rows;
    Mat P = Mat::zeros(2 * n, 12, CV_64F);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            P.at<double>(2 * i + j, 4 * j) = WorldCor.at<double>(i, 0);
            P.at<double>(2 * i + j, 4 * j + 1) = WorldCor.at<double>(i, 1);
            P.at<double>(2 * i + j, 4 * j + 2) = WorldCor.at<double>(i, 2);
            P.at<double>(2 * i + j, 4 * j + 3) = 1;
        }
        for (int j = 0; j < 2; j++)
        {
            P.at<double>(2 * i + j, 8) = -fineC.at<double>(i, j) * WorldCor.at<double>(i, 0);
            P.at<double>(2 * i + j, 9) = -fineC.at<double>(i, j) * WorldCor.at<double>(i, 1);
            P.at<double>(2 * i + j, 10) = -fineC.at<double>(i, j) * WorldCor.at<double>(i, 2);
            P.at<double>(2 * i + j, 11) = -fineC.at<double>(i, j);
        }
    }
    Mat U = P.t() * P;
    vector<double> w;
    Mat V;
    eigen(U, w, V);

    // Find the eigenvector corresponding to the smallest eigenvalue
    int minIndex = min_element(w.begin(), w.end()) - w.begin();
    Mat smallestEigenvector = V.row(minIndex).clone();

    Mat M = Mat::zeros(3, 4, CV_64F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            M.at<double>(i, j) = smallestEigenvector.at<double>(4 * i + j);
        }
    }

    vector<Mat> A(3);
    for (int i = 0; i < 3; i++)
    {
        A[i] = Mat::zeros(1, 3, CV_64F);
        for (int j = 0; j < 3; j++)
        {
            A[i].at<double>(0, j) = M.at<double>(i, j);
        }
    }
    double f = norm(A[2], NORM_L2);
    vector<Mat> R(3);
    for (int i = 0; i < 3; i++)
    {
        R[i] = A[i].clone() / f;
    }

    // Các tham số nội
    double u_0 = Mat(R[0] * R[2].t()).at<double>(0, 0);
    double v_0 = Mat(R[1] * R[2].t()).at<double>(0, 0);
    double beta = sqrtl(Mat(R[1] * R[1].t()).at<double>(0, 0) - v_0 * v_0);
    double skew = 0;
    if (option)
    {
        skew = (Mat(R[1] * R[0].t()).at<double>(0, 0) - u_0 * v_0) / (beta);
    }
    double alpha = sqrtl(Mat(R[0] * R[0].t()).at<double>(0, 0) - u_0 * u_0 - skew * skew);

    Mat K = (Mat_<double>(3, 3) << alpha, skew, u_0, 0, beta, v_0, 0, 0, 1);
    Mat E = K.inv() * M;
    pair<Mat, Mat> result = { K, E };
    return result;
}


int instance = 1;
float bookHeight = 25;

// Creating vector to store vectors of 3D points for each checkerboard image
std::vector<std::vector<cv::Point3f> > objpointsDirect;

// Creating vector to store vectors of 2D points for each checkerboard image
std::vector<std::vector<cv::Point2f> > imgpointsDirect;

void DirectCameraCalibration(Mat& frame)
{
    if (objpointsDirect.size() >= 2)
    {
        Mat WorldCor = Mat::zeros(140, 3, CV_64F);
        Mat fineC = Mat::zeros(140, 2, CV_64F);
        for (int i = 0; i < objpointsDirect.size(); i++)
        {
            for (int j = 0; j < objpointsDirect[i].size(); j++)
            {
                WorldCor.at<double>(i * objpointsDirect[i].size() + j, 0) = objpointsDirect[i][j].x;
                WorldCor.at<double>(i * objpointsDirect[i].size() + j, 1) = objpointsDirect[i][j].y;
                WorldCor.at<double>(i * objpointsDirect[i].size() + j, 2) = objpointsDirect[i][j].z;
            }
        }
        for (int i = 0; i < imgpointsDirect.size(); i++)
        {
            for (int j = 0; j < imgpointsDirect[i].size(); j++)
            {
                fineC.at<double>(i * imgpointsDirect[i].size() + j, 0) = imgpointsDirect[i][j].x;
                fineC.at<double>(i * imgpointsDirect[i].size() + j, 1) = imgpointsDirect[i][j].y;
            }
        }
        cout << "fineC = " << fineC << "\n";
        cout << "WorldCor = " << WorldCor << "\n";
        pair<Mat, Mat> result1 = DirectCalibration(fineC, WorldCor, 0);
        cout << result1.first << "\n";
        pair<Mat, Mat> result2 = DirectCalibration(fineC, WorldCor, 1);
        cout << result2.first << "\n";
        return;
    }
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
        {
            if (instance == 1)
            {
                objp.push_back(cv::Point3f(j * 25, i * 25, bookHeight));
            }
            else
            {
                objp.push_back(cv::Point3f(j * 25, i * 25, 0));
            }
        }
    }

    cv::Mat gray;
    // vector to store the pixel coordinates of detected checker board corners 
    std::vector<cv::Point2f> corner_pts;
    bool success;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_NORMALIZE_IMAGE);


    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
    */
    Mat frame_copy = frame.clone();
    if (success)
    {
        instance++;
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);


        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame_copy, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

        objpointsDirect.push_back(objp);
        imgpointsDirect.push_back(corner_pts);
        cv::imshow("Image", frame_copy);

        std::string name = "image" + std::to_string(objpointsDirect.size()) + ".jpg";
        cv::imwrite(name, frame);
    }
    else
    {
        std::cout << "Khong tim thay hinh vuong!" << std::endl;
        return;
    }
}


