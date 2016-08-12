#include "HoughTransform.h"
#include <iostream>

namespace ht {

	const double M_PI = 3.1415926535897932384626433832795;

	void warpImageByDominantOrientation(cv::Mat& image) {
		double hori, vert;
		ht::getDominantOrientation(image, 0.5, cv::Size(5, 5), hori, vert);

		std::vector<cv::Point2f> srcTri(3);
		std::vector<cv::Point2f> dstTri(3);
		srcTri[0] = cv::Point2f(image.cols / 2.0, image.rows / 2.0);
		srcTri[1] = cv::Point2f(image.cols, image.rows / 2.0 + image.cols / 2.0 * tan(hori / 180.0 * M_PI));
		srcTri[2] = cv::Point2f(image.cols / 2.0 + image.rows / 2.0 / tan(vert / 180.0 * M_PI), image.rows);
		dstTri[0] = cv::Point2f(image.cols / 2.0, image.rows / 2.0);
		dstTri[1] = cv::Point2f(image.cols, image.rows / 2.0);
		dstTri[2] = cv::Point2f(image.cols / 2.0, image.rows);

		cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
		cv::warpAffine(image, image, warpMat, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

		cv::imwrite("result.png", image);
	}

	void houghTransform(const cv::Mat& image, cv::Mat& accum) {
		// convert the image to grayscale
		cv::Mat grayImg;
		if (image.channels() == 1) {
			image.copyTo(grayImg);
		}
		else if (image.channels() == 3) {
			cv::cvtColor(image, grayImg, CV_BGR2GRAY);
		}
		else if (image.channels() == 4) {
			cv::cvtColor(image, grayImg, CV_BGRA2GRAY);
		}

		// blur the image
		cv::blur(grayImg, grayImg, cv::Size(5, 5));

		cv::Mat edgeImg;
		cv::Canny(grayImg, edgeImg, 50, 160, 3);
		//cv::Canny(grayImg, edgeImg, 100, 150, 3);

		cv::imwrite("edge.png", edgeImg);
		
		// compute the accumulation
		double hough_h = ((sqrt(2.0) * (double)(edgeImg.rows > edgeImg.cols ? edgeImg.rows : edgeImg.cols)) / 2.0);

		accum = cv::Mat(hough_h * 2, 180, CV_32F, cv::Scalar(0.0f));
		double center_x = edgeImg.cols / 2.0;
		double center_y = edgeImg.rows / 2.0;
		for (int y = 0; y < edgeImg.rows; ++y) {
			for (int x = 0; x < edgeImg.cols; ++x) {
				if (edgeImg.at<uchar>(y, x) < 250) continue;

				for (int t = 0; t < 180; ++t) {
					double r = ((double)x - center_x) * cos((double)t / 180.0 * M_PI) + ((double)y - center_y) * sin((double)t / 180.0 * M_PI);
					accum.at<float>(r + hough_h, t) += 1.0f;
				}
			}
		}

		//saveImage(accum, "accum.png");
	}

	/**
	 * 指定された画像のエッジを抽出し、水平方向および垂直方向のdominantな角度を返却する。
	 */
	void getDominantOrientation(const cv::Mat& image, float threshold_ratio, const cv::Size& kernel, double& hori, double& vert) {
		cv::Mat accum;
		houghTransform(image, accum);

		visualizeAccum(image, accum, threshold_ratio, "image_edge.png");

		//cv::blur(accum, accum, cv::Size(kernel.width, kernel.height));
		//visualizeAccum(image, accum, "image_edge.png");

		//saveImage(accum, "accum_blur.png");

		vert = getVerticalOrientation(accum, threshold_ratio, kernel);
		hori = getHorizontalOrientation(accum, threshold_ratio, kernel);

		std::cout << "Vert: " << vert << ", Hori: " << hori << std::endl;
	}

	float getVerticalOrientation(const cv::Mat& accum, float threshold_ratio, const cv::Size& kernel) {
		// find teh maximum value
		cv::Mat accum_max;
		cv::reduce(accum, accum_max, 0, CV_REDUCE_MAX);

		float v_max = 0.0f;
		for (int t = 0; t < 180; ++t) {
			if (t > 20 && t < 160) continue;

			if (accum_max.at<float>(0, t) > v_max) {
				v_max = accum_max.at<float>(0, t);
			}
		}

		// change the value below the threshold to 0
		cv::Mat accumV;
		cv::threshold(accum, accumV, v_max * threshold_ratio, 255, cv::THRESH_TOZERO);
		for (int r = 0; r < accumV.rows; ++r) {
			for (int c = 0; c < accumV.cols; ++c) {
				if (c > 20 && c < 160) accumV.at<float>(r, c) = 0;
			}
		}
		//saveImage(accumV, "accumV_th.png");

		// clean up the accum by only keeping the local maximum
		cv::Mat accumV_refined(accumV.size(), CV_32F, cv::Scalar(0.0f));

		for (int r = 0; r < accumV.rows; ++r) {
			for (int t = 0; t < 180; ++t) {
				if (t > 20 && t < 160) continue;
				if (accumV.at<float>(r, t) == 0) continue;

				//Is this point a local maxima (9x9)  
				int max = accumV.at<float>(r, t);
				for (int ly = -4; ly <= 4; ly++) {
					for (int lx = -4; lx <= 4; lx++) {
						if ((ly + r >= 0 && ly + r < accumV.rows) && (lx + t >= 0 && lx + t < accumV.cols)) {
							if (accumV.at<float>(r + ly, t + lx) > max) {
								max = accumV.at<float>(r + ly, t + lx);
								ly = lx = 5;
							}
						}
					}
				}
				if (max > accumV.at<float>(r, t)) continue;

				accumV_refined.at<float>(r, t) += accumV.at<float>(r, t);
				//accumV_refined.at<float>(r, t) += 1.0f;
			}
		}
		//saveImage(accumV_refined, "accumV_refined.png");
		
		// find the theta that has the maximum value
		float vert = 90.0f;
		cv::reduce(accumV_refined, accumV_refined, 0, CV_REDUCE_SUM);
		//saveHistogram(accumV_refined, "accumV_hist.png");
		v_max = 0.0f;
		for (int t = 0; t <= 20; ++t) {
			if (accumV_refined.at<float>(0, t) > v_max) {
				v_max = accumV_refined.at<float>(0, t);
				vert = t + 90;
			}
		}
		for (int t = 160; t < 180; ++t) {
			if (accumV_refined.at<float>(0, t) > v_max) {
				v_max = accumV_refined.at<float>(0, t);
				vert = t - 180 + 90;
			}
		}

		return vert;
	}

	float getHorizontalOrientation(const cv::Mat& accum, float threshold_ratio, const cv::Size& kernel) {
		// find teh maximum value
		cv::Mat accum_max;
		cv::reduce(accum, accum_max, 0, CV_REDUCE_MAX);

		float h_max = 0.0f;
		for (int t = 70; t <= 110; ++t) {
			if (accum_max.at<float>(0, t) > h_max) {
				h_max = accum_max.at<float>(0, t);
			}
		}

		// change the value below the threshold to 0
		cv::Mat accumH;
		cv::threshold(accum, accumH, h_max * threshold_ratio, 255, cv::THRESH_TOZERO);
		for (int r = 0; r < accumH.rows; ++r) {
			for (int c = 0; c < accumH.cols; ++c) {
				if (c < 70 || c > 110) accumH.at<float>(r, c) = 0;
			}
		}
		//saveImage(accumH, "accumH_th.png");

		// clean up the accum by only keeping the local maximum
		cv::Mat accumH_refined(accumH.size(), CV_32F, cv::Scalar(0.0f));

		for (int r = 0; r < accumH.rows; ++r) {
			for (int t = 70; t <= 110; ++t) {
				if (accumH.at<float>(r, t) == 0) continue;

				//Is this point a local maxima (9x9)  
				int max = accumH.at<float>(r, t);
				for (int ly = -4; ly <= 4; ly++) {
					for (int lx = -4; lx <= 4; lx++) {
						if ((ly + r >= 0 && ly + r < accumH.rows) && (lx + t >= 0 && lx + t < accumH.cols)) {
							if (accumH.at<float>(r + ly, t + lx) > max) {
								max = accumH.at<float>(r + ly, t + lx);
								ly = lx = 5;
							}
						}
					}
				}
				if (max > accumH.at<float>(r, t)) continue;

				//accumH_refined.at<float>(r, t) += accumV.at<float>(r, t);
				accumH_refined.at<float>(r, t) += 1.0f;
			}
		}
				
		// find the theta that has the maximum value
		float hori = 0.0f;
		cv::reduce(accumH_refined, accumH_refined, 0, CV_REDUCE_SUM);
		//saveHistogram(accumH_refined, "accumH_hist.png");
		h_max = 0.0f;
		for (int t = 70; t <= 110; ++t) {
			if (accumH_refined.at<float>(0, t) > h_max) {
				h_max = accumH_refined.at<float>(0, t);
				hori = t - 90;
			}
		}

		return hori;
	}

	void saveImage(const cv::Mat& image, const std::string& filename) {
		double minVal;
		double maxVal;
		cv::Point minIdx;
		cv::Point maxIdx;
		cv::minMaxLoc(image, &minVal, &maxVal, &minIdx, &maxIdx);

		cv::imwrite(filename.c_str(), (image - minVal) / (maxVal - minVal) * 255);
		//cv::imwrite(filename.c_str(), image / maxVal * 255);
	}

	void saveHistogram(const cv::Mat& mat, const std::string& filename) {
		cv::Mat mat2 = mat.clone();

		cv::Mat mat_max;
		cv::reduce(mat, mat_max, 0, CV_REDUCE_MAX);
		cv::reduce(mat_max, mat_max, 1, CV_REDUCE_MAX);

		//std::cout << "Max: " << mat_max.at<float>(0, 0) << std::endl;

		mat2 *= (100.0 / mat_max.at<float>(0, 0));

		if (mat2.cols == 1) {
			mat2 = mat2.t();
		}

		cv::Mat hist(100, mat2.cols, CV_8U, cv::Scalar(255));
		for (int c = 0; c < mat2.cols; ++c) {
			if (mat2.at<float>(0, c) > 0) {
				cv::line(hist, cv::Point(c, hist.rows - 1), cv::Point(c, hist.rows - 1 - mat2.at<float>(0, c)), cv::Scalar(0), 1);
			}
		}

		cv::imwrite(filename.c_str(), hist);
	}

	void visualizeAccum(const cv::Mat& image, const cv::Mat& accum, float threshold_ratio, const std::string& filename) {
		cv::Mat accum_max;
		cv::reduce(accum, accum_max, 0, CV_REDUCE_MAX);
		cv::reduce(accum_max, accum_max, 1, CV_REDUCE_MAX);

		cv::Mat accum_threshold;
		cv::threshold(accum, accum_threshold, accum_max.at<float>(0, 0) * threshold_ratio, 255, cv::THRESH_TOZERO);

		cv::Mat result = image.clone();

		for (int r = 0; r < accum_threshold.rows; ++r) {
			for (int t = 0; t < accum_threshold.cols; ++t) {
				if ((t > 20 && t < 70) || (t > 110 && t < 160)) continue;

				if (accum_threshold.at<float>(r, t) > 0) {
					//Is this point a local maxima (9x9)  
					int max = accum_threshold.at<float>(r, t);
					for (int ly = -4; ly <= 4; ly++) {
						for (int lx = -4; lx <= 4; lx++) {
							if ((ly + r >= 0 && ly + r < accum_threshold.rows) && (lx + t >= 0 && lx + t < accum_threshold.cols)) {
								if (accum_threshold.at<float>(r + ly, t + lx) > max) {
									max = accum_threshold.at<float>(r + ly, t + lx);
									ly = lx = 5;
								}
							}
						}
					}
					if (max > accum_threshold.at<float>(r, t)) continue;

					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					if (t >= 45 && t <= 135) {
						x1 = 0;
						y1 = ((double)(r - (accum_threshold.rows / 2)) - ((x1 - (image.cols / 2)) * cos(t / 180.0 * M_PI))) / sin(t / 180.0 * M_PI) + (image.rows / 2);
						x2 = image.cols;
						y2 = ((double)(r - (accum_threshold.rows / 2)) - ((x2 - (image.cols / 2)) * cos(t / 180.0 * M_PI))) / sin(t / 180.0 * M_PI) + (image.rows / 2);
					}
					else {
						y1 = 0;
						x1 = ((double)(r - (accum_threshold.rows / 2)) - ((y1 - (image.rows / 2)) * sin(t / 180.0 * M_PI))) / cos(t / 180.0 * M_PI) + (image.cols / 2);
						y2 = image.rows;
						x2 = ((double)(r - (accum_threshold.rows / 2)) - ((y2 - (image.rows / 2)) * sin(t / 180.0 * M_PI))) / cos(t / 180.0 * M_PI) + (image.cols / 2);
					}
					cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
				}
			}
		}

		cv::imwrite(filename.c_str(), result);
	}
}
