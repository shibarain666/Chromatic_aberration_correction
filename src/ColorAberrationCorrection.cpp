#include <vector>
#include <opencv2\opencv.hpp>
#include "ColorAberrationCorrection.h"

void rmCA(std::vector<cv::Mat>& bgrVec, int threshold, std::string direction) {

	if (!strcmp(direction.c_str(), "vertical")) {
		//transpose the B, G, R channel image to correct chromatic aberration in vertical direction 
		bgrVec[0] = bgrVec[0].t();
		bgrVec[1] = bgrVec[1].t();
		bgrVec[2] = bgrVec[2].t();
	}

	int height = bgrVec[0].rows;
	int width = bgrVec[0].cols;

	/* valid_map are only use to validate algorithm */
	cv::Mat valid_map(height, width, CV_8UC3);
	cv::merge(bgrVec, valid_map);

	/* RGB -> XYZ */
	cv::Mat cie_xyz_map(height, width, CV_32FC3);
	cvtColor(valid_map, cie_xyz_map, cv::COLOR_BGR2XYZ);

	for (int i = 0; i < height; ++i) {

		uchar* bptr = bgrVec[0].ptr<uchar>(i);
		uchar* gptr = bgrVec[1].ptr<uchar>(i);
		uchar* rptr = bgrVec[2].ptr<uchar>(i);

		cv::Vec3b* ptr = cie_xyz_map.ptr<cv::Vec3b>(i);

		for (int j = 1; j < width - 1; ++j)
		{
			//find the edge by finding green channel gradient bigger than threshold
			if (abs(gptr[j + 1] - gptr[j - 1]) >= threshold)
			{
				// determine +/- sign of this edge
				int sign = 0;
				if (gptr[j + 1] - gptr[j - 1] > 0) { sign = 1; }
				else { sign = -1; }

				//Searching the boundary for correction range(lpos = left position, rpos = right position)
				//The transition region is the union of the pixels where at least one of the color values varies
				int lpos = j - 1, rpos = j + 1;
				//searching left side
				for (; lpos > 0; --lpos){
					//make sure the gradient is the same sign with edge
					int ggrad = (gptr[lpos + 1] - gptr[lpos - 1]) * sign;
					int bgrad = (bptr[lpos + 1] - bptr[lpos - 1]) * sign;
					int rgrad = (rptr[lpos + 1] - rptr[lpos - 1]) * sign;
					if (std::max(std::max(bgrad, ggrad), rgrad) < threshold) { break; }
				}

				//searching right side
				for (; rpos < width - 1; ++rpos){
					//make sure the gradient is the same sign with edge
					int ggrad = (gptr[rpos + 1] - gptr[rpos - 1]) * sign;
					int bgrad = (bptr[rpos + 1] - bptr[rpos - 1]) * sign;
					int rgrad = (rptr[rpos + 1] - rptr[rpos - 1]) * sign;
					if (std::max(std::max(bgrad, ggrad), rgrad) < threshold) { break; }
				}

				for (int k = lpos; k <= rpos; ++k) {

					double x = (double)ptr[k][0] / ((double)ptr[k][0] + (double)ptr[k][1] + (double)ptr[k][2]);
					double y = (double)ptr[k][1] / ((double)ptr[k][0] + (double)ptr[k][1] + (double)ptr[k][2]);

					int pf_lpos;
					int pf_rpos;

					/* if it is in a certain CIE plot area */
					if ((y <= 1.3692 * x - 0.0927) && (y <= -0.2048 * x + 0.393) && (y >= 0.0551 * x - 0.0227)) {
						pf_lpos = k;
					}
					else {
						continue;
					}
					for (pf_rpos = k; pf_rpos <= rpos; ++pf_rpos) {
						x = (double)ptr[pf_rpos][0] / ((double)ptr[pf_rpos][0] + (double)ptr[pf_rpos][1] + (double)ptr[pf_rpos][2]);
						y = (double)ptr[pf_rpos][1] / ((double)ptr[pf_rpos][0] + (double)ptr[pf_rpos][1] + (double)ptr[pf_rpos][2]);
						if ((y > 1.3692 * x - 0.0927) || (y > -0.2048 * x + 0.393) || (y < 0.0551 * x - 0.0227)) {
							break;
						}
					}
					/* key point, 5 seems the best */
					pf_lpos -= 5;
					pf_rpos += 5;
					if (pf_rpos > width - 1) {
						pf_rpos = width - 1;
					}
					if (pf_lpos < 0) {
						pf_lpos = 0;
					}

					int gbdiff_lpos = gptr[pf_lpos] - bptr[pf_lpos];
					int grdiff_lpos = gptr[pf_lpos] - rptr[pf_lpos];
					int gbdiff_rpos = gptr[pf_rpos] - bptr[pf_rpos];
					int grdiff_rpos = gptr[pf_rpos] - rptr[pf_rpos];

					//record the maximum and minimum color difference between R&G and B&G of range boundary
					int gbmaxVal = std::max(gbdiff_lpos, gbdiff_rpos);
					int gbminVal = std::min(gbdiff_lpos, gbdiff_rpos);
					int grmaxVal = std::max(grdiff_lpos, grdiff_rpos);
					int grminVal = std::min(grdiff_lpos, grdiff_rpos);

					for (int p = pf_lpos; p <= pf_rpos; ++p) {

						int k_gr = gptr[p] - rptr[p];
						int k_gb = gptr[p] - bptr[p];

						bptr[p] = cv::saturate_cast<uchar>(k_gb > gbmaxVal ? gptr[p] - gbmaxVal :
							(k_gb < gbminVal ? gptr[p] - gbminVal : bptr[p]));
						rptr[p] = cv::saturate_cast<uchar>(k_gr > grmaxVal ? gptr[p] - grmaxVal :
							(k_gr < grminVal ? gptr[p] - grminVal : rptr[p]));
					}
					k = pf_rpos + 1;
				}
				j = rpos - 1;
			}
		}
	}

	if (!strcmp(direction.c_str(), "vertical")) {
		//rotate the image back to original position in vertical situation.
		bgrVec[0] = bgrVec[0].t();
		bgrVec[1] = bgrVec[1].t();
		bgrVec[2] = bgrVec[2].t();
	}
}

void CACorrection(cv::Mat& Src, cv::Mat& Dst, int threshold)
{
	std::vector<cv::Mat> bgrVec(3);
	//split the color image into individual color channel for convenient in calculation
	cv::split(Src, bgrVec);

	//setting threshold to find the edge and correction range(in g channel)
	rmCA(bgrVec, threshold, "horizontal");
	rmCA(bgrVec, threshold, "vertical");

	cv::merge(bgrVec, Dst);
}
