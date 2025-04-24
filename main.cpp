#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include "object_utils.h"
#include "calibration_utils.h"
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>

using namespace cv;
using namespace std;

vector<uchar> mjpegBuffer;
bool stopStreaming = false;
mutex bufferMutex;  // Bảo vệ mjpegBuffer

// Hàm callback của libcurl khi nhận dữ liệu từ stream
size_t curlWriteCallback(void* ptr, size_t size, size_t nmemb, void* userdata) {
	size_t totalSize = size * nmemb;
	auto* buffer = reinterpret_cast<vector<uchar>*>(userdata);
	lock_guard<mutex> lock(bufferMutex);  // Khóa khi ghi
	buffer->insert(buffer->end(), (uchar*)ptr, (uchar*)ptr + totalSize);
	return totalSize;
}

// Hàm xử lý luồng MJPEG từ ESP32-CAM
void streamMJPEG(const string& url) {
	CURL* curl = curl_easy_init();
	if (!curl) {
		cerr << "Khởi tạo CURL thất bại!\n";
		return;
	}

	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &mjpegBuffer);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
	curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024 * 1024);
	curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

	CURLcode res = curl_easy_perform(curl);
	if (res != CURLE_OK) {
		cerr << "Loi CURL: " << curl_easy_strerror(res) << endl;
	}

	curl_easy_cleanup(curl);
}

Mat frame;

int stage = 5;
inline void processStreamingFrame()
{
	imshow("ESP32-CAM Stream", frame);
	int key = waitKey(1);
	if (stage == 1)
	{
		cout << "Press Enter to start Object Detection or ESC to exit.\n";
		stage++;
	}
	if (stage == 2)
	{
		if (key == 13)
		{
			ObjectDetection(frame, 1);
			destroyWindow("Object Detection");
			stage++;
			cout << "Press Enter to start Chessboard Calibration or ESC to exit.\n";
			return;
		}
		else
		{
			ObjectDetection(frame, 0);
		}
	}
	else if (stage == 3)
	{
		if (key == 13)
		{
			if (ChessBoardCalibration(frame))
			{
				stage++;
				destroyWindow("Chessboard Calibration");
				cout << "Press Enter to Calculate Object Length or ESC to exit.\n";
				return;
			}
		}
	}
	else if (stage == 4)
	{
		if (key == 13)
		{
			cout << ObjectLength() << "\n";
			stage++;
			cout << "Press Enter to start Direct Camera Calibration or ESC to exit.\n";
			return;
		}
	}
	else if(stage == 5)
	{
		if (key == 13)
		{
			if (DirectCameraCalibration(frame))
			{
				stage++;
				stopStreaming = true;
			}
		}
	}
	else
	{
		exit(0);
	}
	if (key == 27) { // ESC
		stopStreaming = true;
	}
}


int main() {
	thread streamThread(streamMJPEG, "http://192.168.1.200/");

	namedWindow("ESP32-CAM Stream", WINDOW_AUTOSIZE);
	vector<uchar> localBuffer;

	vector<uchar> jpegStart = { 0xFF, 0xD8 };
	vector<uchar> jpegEnd = { 0xFF, 0xD9 };

	while (!stopStreaming) {
		this_thread::sleep_for(chrono::milliseconds(10));

		// Copy an toàn mjpegBuffer
		{
			lock_guard<mutex> lock(bufferMutex);
			localBuffer = mjpegBuffer;
		}

		// Tìm vị trí bắt đầu và kết thúc JPEG
		auto startIt = search(localBuffer.begin(), localBuffer.end(), jpegStart.begin(), jpegStart.end());
		auto endIt = search(startIt, localBuffer.end(), jpegEnd.begin(), jpegEnd.end());

		if (startIt != localBuffer.end() && endIt != localBuffer.end() && endIt > startIt) {
			endIt += jpegEnd.size(); // Bao gồm cả 0xFFD9
			vector<uchar> jpeg(startIt, endIt);

			// Cập nhật buffer chính
			size_t eraseLen = distance(localBuffer.begin(), endIt);
			{
				lock_guard<mutex> lock(bufferMutex);
				if (eraseLen <= mjpegBuffer.size()) {
					mjpegBuffer.erase(mjpegBuffer.begin(), mjpegBuffer.begin() + eraseLen);
				}
			}

			// Decode và hiển thị
			frame = imdecode(jpeg, IMREAD_COLOR);
			if (!frame.empty()) {
				processStreamingFrame();
			}
		}
	}

	destroyAllWindows();
	streamThread.join();
	return 0;
}
