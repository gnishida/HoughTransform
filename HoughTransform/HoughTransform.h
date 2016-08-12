#pragma once

#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace ht {

	void warpImageByDominantOrientation(cv::Mat& image);
	void houghTransform(const cv::Mat& image, cv::Mat& accum);
	void getDominantOrientation(const cv::Mat& image, float threshold_ratio, const cv::Size& kernel, double& hori, double& vert);
	float getVerticalOrientation(const cv::Mat& accum, float threshold_ratio, const cv::Size& kernel);
	float getHorizontalOrientation(const cv::Mat& accum, float threshold_ratio, const cv::Size& kernel);
	void saveImage(const cv::Mat& image, const std::string& filename);
	void saveHistogram(const cv::Mat& mat, const std::string& filename);

	void visualizeAccum(const cv::Mat& image, const cv::Mat& accum, float threshold_ratio, const std::string& filename);
}
