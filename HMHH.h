#pragma once
#include <opencv2\opencv.hpp>
#include "svm.h"
using namespace cv;
class HMHH
{
public:
	HMHH(int);
	HMHH();
	~HMHH();
	float* get_feature(Mat);
	int get_feature_len();
	int get_result(Mat);

private:
	void update_window(Mat);
	void update_feature(CvMat**);
	int detect_window_len;
	int img_width;
	int img_hight;
	Mat* last_frame;
	Mat* current_frame;
	CvMat* MHI_feature;
	CvMat** Diff;
	CvMat I;
	CvMat** HMHH_feature;
	int HMHH_len;
	CvMat** window;
	bool predict;
	void init_predict();
	svm_model* svmModel;
	float* feature_max;
	float* feature_min;
};