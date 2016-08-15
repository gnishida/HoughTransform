#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "HoughTransform.h"
#include <iostream>

int main() {
	int NUM_DATA = 26;

	/*
	cv::Mat img = cv::imread("../testdata/image12.png");
	ht::warpImageByDominantOrientation(img, 0, 0);
	cv::imwrite("result.png", img);
	*/
	
	std::vector<float> horis = { -1, 0, -4, 0, 3, 3, 1, 3, -1, 0, 2, 0, 0, 0, 0, 0, 
								0, 0, 0, 2, 1, 0, 0, 0, 1, 0 };
	std::vector<float> verts = { 88, 91, 92, 90, 98, 94, 90, 90, 90, 85, 90, 89, 90, 90, 90, 88, 
								88, 90, 90, 86, 90, 90, 88, 87, 88, 90 };

	float error = 0.0f;
	for (int i = 0; i < NUM_DATA; ++i) {
		char name[256];
		sprintf(name, "../testdata/image%d.png", i + 1);

		cv::Mat img = cv::imread(name);
		error += ht::warpImageByDominantOrientation(img, horis[i], verts[i]);

		char name2[256];
		sprintf(name2, "../results/result%d.png", i + 1);
		cv::imwrite(name2, img);
	}

	std::cout << "Avg error: " << error / NUM_DATA << std::endl;

	return 0;
}