#include <iostream>
#include <opencv2\opencv.hpp>
#include "ColorAberrationCorrection.h"

int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cout << "Too few input arguments." << std::endl;
		std::cout << "Example: .\\chromatic_aberration.exe .\\imgs\\purple_fringe_tree.jpg 30" << std::endl;
		return -1;
	}

	cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat dst;

	int threshold = 30; //default
	if (argc == 3) {
		threshold = atoi(argv[2]);
	}

	std::cout << "Threshold: " << threshold << std::endl;
	CACorrection(src, dst, threshold);

	cv::imwrite(".\\imgs\\result.bmp", dst);

	ShowManyImages("Comparison", 2, src, dst);
	cv::destroyAllWindows();
	return 0;
}
