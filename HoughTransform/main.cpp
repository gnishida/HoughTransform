#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "HoughTransform.h"

int main() {
	cv::Mat img = cv::imread("image6.png");
	
	/*
	std::vector<cv::Point2f> srcTri(3);
	std::vector<cv::Point2f> dstTri(3);
	srcTri[0] = cv::Point2f(0, 0);
	srcTri[1] = cv::Point2f(100, 0);
	srcTri[2] = cv::Point2f(0, 100);
	dstTri[0] = cv::Point2f(0, 0);
	dstTri[1] = cv::Point2f(100, -10);
	dstTri[2] = cv::Point2f(-10, 100);
	cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
	cv::warpAffine(img, img, warpMat, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	cv::imwrite("facade1_warped.png", img);
	*/

	ht::warpImageByDominantOrientation(img);

	return 0;
}