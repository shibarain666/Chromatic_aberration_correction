#ifndef _CA_H_

#include <opencv2\opencv.hpp>
#include <string>

void rmCA(std::vector<cv::Mat>& bgrVec, int threshold, std::string direction);

void CACorrection(cv::Mat& Src, cv::Mat& Dst, int threshold);

void ShowManyImages(std::string title, int nArgs, ...);

#endif // !_CA_H_
