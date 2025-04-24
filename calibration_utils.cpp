#pragma once
#include <opencv2/opencv.hpp>
#include "calibration_utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 10, 7 };

// Creating vector to store vectors of 3D points for each checkerboard image
vector<vector<Point3f>> objpoints;

// Creating vector to store vectors of 2D points for each checkerboard image
vector<vector<Point2f>> imgpoints;

Mat intrinsicMatrix, rotationMatrix, translationVector;


bool ChessBoardCalibration(Mat& frame)
{
    vector<Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
        {
            objp.push_back(Point3f(float(j * 25), float(i * 25), 0));
        }
    }

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Point2f> corner_pts;
    bool success;

    success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, 
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);

	Mat frame_copy = frame.clone();
    if (success)
    {
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);


        // refining pixel coordinates for given 2d points.
        cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);

        // Displaying the detected corner points on the checker board
        drawChessboardCorners(frame_copy, Size(CHECKERBOARD[0], CHECKERBOARD[1]), 
            corner_pts, success);

        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
        imshow("Chessboard Calibration", frame_copy);

        if (!filesystem::exists("ChessboardCalibration"))
        {
            filesystem::create_directory("ChessboardCalibration");
        }

        string name = "ChessboardCalibration/image " + to_string(objpoints.size()) + ".jpg";
        imwrite(name, frame);
		cout << "Da luu hinh thu " << to_string(objpoints.size()) << "\n";
    }   
    else
    {
		cout << "Khong tim thay hinh vuong!" << "\n";
        return 0;
    }
    if (objpoints.size() >= 15)
    {
        Mat cameraMatrix, distCoeffs;
        vector<Mat> rvecs, tvecs;

        calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, rvecs, tvecs);

        ofstream logfile("ChessBoardCalibration.txt", ios::out);  // Mở file ở chế độ overwrite
        logfile << "cameraMatrix : " << cameraMatrix << "\n";
		intrinsicMatrix = cameraMatrix.clone();
        logfile << "distCoeffs : " << distCoeffs << "\n";
        for (size_t i = 0; i < rvecs.size(); ++i) {
            logfile << "Image " << i + 1 << "\n";

            // Convert rotation vector to rotation matrix
            Mat R;
            Rodrigues(rvecs[i], R);

            logfile << "Rotation Matrix:\n" << R << "\n";
            logfile << "Translation Vector:\n" << tvecs[i] << "\n";
			logfile << "------------------------\n";
        }
        Mat R;
		Rodrigues(rvecs[0], R);
		rotationMatrix = R.clone();
		translationVector = tvecs[0].clone();
		logfile.close();
		cout << "Da luu ket qua vao file ChessBoardCalibration.txt\n";
        return 1;
    }
    else
    {
        return 0;
    }
}

pair<Mat, Mat> DirectCalibration(Mat& fineC, Mat& WorldCor)
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

	// Tìm giá trị riêng nhỏ nhất
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
    double skew = (Mat(R[1] * R[0].t()).at<double>(0, 0) - u_0 * v_0) / (beta);
    double alpha = sqrtl(Mat(R[0] * R[0].t()).at<double>(0, 0) - u_0 * u_0 - skew * skew);

    Mat K = (Mat_<double>(3, 3) << alpha, skew, u_0, 0, beta, v_0, 0, 0, 1);
    Mat E = K.inv() * M;
    pair<Mat, Mat> result = { K, E };
    return result;
}


int instance = 1;
float bookHeight = 25;

vector<vector<Point3f>> objpointsDirect;

vector<vector<Point2f>> imgpointsDirect;

bool DirectCameraCalibration(Mat& frame)
{
    vector<Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
        {
            if (instance == 1)
            {
                objp.push_back(Point3f(j * 25, i * 25, bookHeight));
            }
            else
            {
                objp.push_back(Point3f(j * 25, i * 25, 0));
            }
        }
    }

    Mat gray;
    vector<Point2f> corner_pts;
    bool success;

    cvtColor(frame, gray, COLOR_BGR2GRAY);
 
    success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, 
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);

    Mat frame_copy = frame.clone();
    if (success)
    {
        instance++;
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);


        cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);

        drawChessboardCorners(frame_copy, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

        objpointsDirect.push_back(objp);
        imgpointsDirect.push_back(corner_pts);
        imshow("Direct Calibration", frame_copy);

        if (!filesystem::exists("DirectCalibration")) 
        {
            filesystem::create_directory("DirectCalibration");
        }

        string name = "DirectCalibration/image " + to_string(objpointsDirect.size()) + ".jpg";
        imwrite(name, frame);
        cout << "Da luu hinh thu " << to_string(objpointsDirect.size()) << "\n";
    }
    else
    {
        cout << "Khong tim thay hinh vuong!" << "\n";
        return 0;
    }
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

        ofstream logfile("DirectCalibration.txt", ios::out);  // Mở file ở chế độ overwrite
        logfile << "fineC = " << fineC << "\n";
        logfile << "WorldCor = " << WorldCor << "\n";
		logfile << "Ket qua direct calibration:" << "\n";
        pair<Mat, Mat> result = DirectCalibration(fineC, WorldCor);
        logfile << "K = " << result.first << "\n";
        logfile << "E = " << result.second << "\n";
        logfile.close();

		cout << "Da luu ket qua vao file DirectCalibration.txt\n";

        return 1;
    }
    else
    {
        return 0;
    }
}


