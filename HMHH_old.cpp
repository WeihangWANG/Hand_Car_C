// HMHH.cpp: 定义控制台应用程序的入口点。
//
#include <iostream>
#include <fstream>
#include "HMHH.h"
#include <algorithm> 
#include <time.h>

using namespace cv;
using namespace std;


HMHH::HMHH()
{	
	detect_window_size = 20;
	img_width = 120;
	img_hight = 120;
	HMHH_len = 5;
	MHI_feature = new CvMat*[detect_window_size];
	HMHH_feature = new CvMat*[HMHH_len];
	for (int i = 0; i < detect_window_size; i++) {
		MHI_feature[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(MHI_feature[i]);
	}
	for (int i = 0; i < HMHH_len; i++) {
		HMHH_feature[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(HMHH_feature[i]);
	}
	I = *cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(&I);
	window = new CvMat*[detect_window_size];
	for (int j = 0; j < detect_window_size; j++) {
		window[j] = cvCreateMat(120, 120, CV_8UC1);
		cvSetZero(window[j]);
	}
	predict = false;
}


HMHH::~HMHH()
{
	delete[] MHI_feature;
	delete[] HMHH_feature;
	delete[] window;
}

void HMHH::update_window(Mat src){
		Mat u_src = Mat(Size(120, 120), CV_8UC1);
		resize(src, u_src, Size(120, 120));
		for (int j = 0; j < detect_window_size -1; j++) window[j] = cvCloneMat(window[j + 1]);
		window[detect_window_size -1] = cvCloneMat(&(CvMat)u_src);
		update_feature(window);
	}



void HMHH::update_feature(CvMat** window) {
	//calculate D(u,v)
	Mat D(img_width, img_hight, CV_64FC1);
	Mat temp = cvarrToMat(window[detect_window_size - 2]);
	last_frame = &temp;
	Mat temp2 = cvarrToMat(window[detect_window_size - 1]);
	current_frame = &temp2;
	absdiff(*last_frame, *current_frame, D);
	threshold(D, D, 25, 255, THRESH_BINARY);
	for (int i = 0; i < detect_window_size-1; i++)
		MHI_feature[i] = cvCloneMat(MHI_feature[i + 1]);
	for (int u = 0; u < img_hight; u++)
		for (int v = 0; v < img_width; v++) {
			if (D.data[u*D.cols*D.elemSize() + v] == 255){
				cvmSet(MHI_feature[detect_window_size - 1], u, v, 255);
				cvmSet(&I, u, v, cvmGet(&I, u, v) + 1);
			}
			else{
				int val = std::max(0.0, cvmGet(MHI_feature[detect_window_size - 2], u, v) - 1);
				cvmSet(MHI_feature[detect_window_size - 1], u, v, val);
				int pattern_num = cvmGet(&I, u, v);
				if (pattern_num > 0 && pattern_num <= HMHH_len) {
					//std::cout << pattern_num << std::endl;
					cvmSet(HMHH_feature[pattern_num - 1], u, v, cvmGet(HMHH_feature[pattern_num - 1], u, v)+1);
				}
				cvmSet(&I, u, v, 0);
			}
		}
}

int HMHH::get_feature_len(){
	return 255 + HMHH_len*(img_hight + img_width);
}

float* HMHH::get_feature(Mat src) {	
	update_window(src);
	float* array = new float[255 + HMHH_len*(img_hight + img_width)];

	Mat hist(255,1,CV_16UC1);
	const int histSize = 255;
	const int channels = 0;
	float range[] = { 1, 256 };
	const float *ranges[] = { range };
	Mat temp = cvarrToMat(MHI_feature[detect_window_size - 1]);
	temp.convertTo(temp, CV_8UC1);

	calcHist(&temp, 1, &channels, Mat(), hist, 1, &histSize, &ranges[0],true, false);

	memcpy(array, hist.data, 255 * sizeof(float));
	
	for (int i = 0; i < HMHH_len; i++) {
		Mat MHH_b,MHH_b_u,MHH_b_v;
		Mat temp = cvarrToMat(HMHH_feature[i]);
		temp.convertTo(temp, CV_8UC1);
		threshold(temp, MHH_b, 0, 1, THRESH_BINARY);
		reduce(MHH_b, MHH_b_u, 0, CV_REDUCE_SUM, CV_32F);
		reduce(MHH_b, MHH_b_v, 1, CV_REDUCE_SUM, CV_32F);
		memcpy(array+255+i*(img_hight + img_width), MHH_b_u.data, img_width * sizeof(float));
		memcpy(array + 255 + i*(img_hight + img_width)+ img_width, MHH_b_v.data, img_hight * sizeof(float));
	}
	return array;
}

void HMHH::init_predict() {
	/*initialize svmModel*/
	string modelPath = "";
	string scalePath = "";
	svmModel = svm_load_model(modelPath.c_str());

	/*initialize feature_max & feature_min*/
	std::ifstream feature_scale;
	int len = get_feature_len();
	feature_max = (float *)malloc(len * sizeof(float));
	feature_min = (float *)malloc(len * sizeof(float));
	feature_scale.open(scalePath.c_str());
	while (!feature_scale.eof())
	{	
		int index;
		feature_scale >> index;
		feature_scale >> feature_max[index];
		feature_scale >> feature_min[index];
	}
	
	feature_scale.close();

	predict = true;
}

int HMHH::get_result(Mat src) {
	float* feature = get_feature(src);
	 
	/* initialize settings for predict*/
	if (!predict) {
		init_predict();
	}
	int len = get_feature_len();
	svm_node *x;
	x = (svm_node *)malloc((len+1) * sizeof(svm_node));
	/*rescale feature and construct svmnode*/
	int index = 0;
	for (int i=0 ; i < len; i++)
	{	
		if (feature_max[i] == feature_min[i]) continue;
		feature[i] = -1 + 2 * (feature[i] - feature_min[i]) / (feature_max[i] - feature_min[i]);
		if (feature[i]!=0){
			x[index].index = i;
			x[index].value = feature[i];
			++index;
		}
	}
	x[index].index = -1;

	return svm_predict(svmModel, x);

}