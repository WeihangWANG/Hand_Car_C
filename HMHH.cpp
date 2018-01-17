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
	detect_window_len = 4;
	img_width = 120;
	img_hight = 120;
	HMHH_len = 5;
	MHI_feature = cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(MHI_feature);
	HMHH_feature = new CvMat*[HMHH_len];
	for (int i = 0; i < HMHH_len; i++) {
		HMHH_feature[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(HMHH_feature[i]);
	}
	Diff = new CvMat*[detect_window_len];
	for (int i = 0; i < detect_window_len; i++) {
		Diff[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(Diff[i]);
	}
	I = *cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(&I);
	window = new CvMat*[detect_window_len];
	for (int j = 0; j < detect_window_len; j++) {
		window[j] = cvCreateMat(120, 120, CV_8UC1);
		cvSetZero(window[j]);
	}
	predict = false;
}

HMHH::HMHH(int len)
{
	detect_window_len = len;
	img_width = 120;
	img_hight = 120;
	HMHH_len = 5;
	MHI_feature = cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(MHI_feature);
	HMHH_feature = new CvMat*[HMHH_len];
	for (int i = 0; i < HMHH_len; i++) {
		HMHH_feature[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(HMHH_feature[i]);
	}
	Diff = new CvMat*[detect_window_len];
	for (int i = 0; i < detect_window_len; i++) {
		Diff[i] = cvCreateMat(img_width, img_hight, CV_64FC1);
		cvSetZero(Diff[i]);
	}
	I = *cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(&I);
	window = new CvMat*[detect_window_len];
	for (int j = 0; j < detect_window_len; j++) {
		window[j] = cvCreateMat(120, 120, CV_8UC1);
		cvSetZero(window[j]);
	}
	predict = false;
}


HMHH::~HMHH()
{
	/*delete &I;
	delete[] MHI_feature;
	delete[] HMHH_feature;
	delete[] window;*/
}

void HMHH::update_window(Mat src){
		Mat u_src = Mat(Size(120, 120), CV_8UC1);
		resize(src, u_src, Size(120, 120));
		for (int j = 0; j < detect_window_len - 1; j++) {
			cvReleaseMat(&window[j]);
			window[j] = NULL;
			window[j] = cvCloneMat(window[j + 1]);
		}
		cvReleaseMat(&window[detect_window_len - 1]);
		window[detect_window_len - 1] = NULL;
		window[detect_window_len -1] = cvCloneMat(&(CvMat)u_src);
		update_feature(window);
	}



void HMHH::update_feature(CvMat** window) {
	//calculate D(u,v)
	Mat D(img_width, img_hight, CV_64FC1);
	Mat temp = cvarrToMat(window[detect_window_len - 2]);
	last_frame = &temp;
	Mat temp2 = cvarrToMat(window[detect_window_len - 1]);
	current_frame = &temp2;
	absdiff(*last_frame, *current_frame, D);
	threshold(D, D, 25, 255, THRESH_BINARY);
	int i;
	for (i = 0; i < detect_window_len-1; i++)
	{
		cvReleaseMat(&Diff[i]); 
		Diff[i] = NULL;
		Diff[i] = cvCloneMat(Diff[i + 1]);
	}
	
	for (int u = 0; u < img_hight; u++)
		for (int v = 0; v < img_width; v++) {
			if (D.data[u*D.cols*D.elemSize() + v] == 255){
				cvmSet(MHI_feature, u, v, 255);
				cvmSet(&I, u, v, cvmGet(&I, u, v) + 1);
			}
			else{
				int val = std::max(0.0, cvmGet(MHI_feature, u, v) - 255/detect_window_len);
				cvmSet(MHI_feature, u, v, val);
			}
		}
	D.convertTo(D, CV_64FC1);
	cvReleaseMat(&Diff[i]);
	Diff[i] = NULL;
	Diff[i] = cvCloneMat(&(CvMat)D);
}

int HMHH::get_feature_len(){
	return 255 + HMHH_len*(img_hight + img_width);
}

float* HMHH::get_feature(Mat src) {	
	update_window(src);
	
	I = *cvCreateMat(img_width, img_hight, CV_64FC1);
	cvSetZero(&I);

	for (int i = 0; i < HMHH_len; i++) {
		cvSetZero(HMHH_feature[i]);
	}

	for (int i = 0;i<detect_window_len;i++)
		for (int u = 0; u < img_hight; u++)
			for (int v = 0; v < img_width; v++) {
				if (cvmGet(Diff[i],u,v) == 255) {
					cvmSet(&I, u, v, cvmGet(&I, u, v) + 1);
				}
				else {
					int pattern_num = cvmGet(&I, u, v);
					if (pattern_num > 0 && pattern_num <= HMHH_len) {
						cvmSet(HMHH_feature[pattern_num - 1], u, v, cvmGet(HMHH_feature[pattern_num - 1], u, v) + 1);
					}
					cvmSet(&I, u, v, 0);
				}
			}

	/*imshow("MHI", (Mat)MHI_feature);
	imshow("HMHH1", (Mat)HMHH_feature[0]);
	imshow("HMHH2", (Mat)HMHH_feature[1]);
	imshow("HMHH3", (Mat)HMHH_feature[2]);
	imshow("HMHH4", (Mat)HMHH_feature[3]);
	imshow("HMHH5", (Mat)HMHH_feature[4]);
	waitKey(0);*/


	float* array = new float[255 + HMHH_len*(img_hight + img_width)];

	Mat hist(255,1,CV_16UC1);
	const int histSize = 255;
	const int channels = 0;
	float range[] = { 1, 256 };
	const float *ranges[] = { range };
	Mat temp = cvarrToMat(MHI_feature);
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
	string modelPath = "svm_model.model";
	string scalePath = "feature_scale.txt";
	svmModel = svm_load_model(modelPath.c_str());

	/*initialize feature_max & feature_min*/
	std::ifstream feature_scale;
	int len = get_feature_len();
	feature_max = (float *)malloc(len * sizeof(float));
	memset(feature_max, 0, len * sizeof(float));
	feature_min = (float *)malloc(len * sizeof(float));
	memset(feature_min, 0, len * sizeof(float));
	feature_scale.open(scalePath.c_str());
	while (!feature_scale.eof())
	{	
		int index;
		feature_scale >> index;
		feature_scale >> feature_min[index-1];
		feature_scale >> feature_max[index-1];
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

//int main(int argc, char** argv) {
//	string path, ofilename;// = "C:\\Users\\adonis\\Documents\\HMHH\\BMWcuttdata\\action01\\d_a01_s01_e01_s_depth";
//	int start, end, window_len=0;
//	HMHH m;
//
//
//	path = argv[1];
//	ofilename = argv[2];
//	start = atoi(argv[3]);
//	end = atoi(argv[4]);
//
//	if (argc > 5) {
//		window_len = atoi(argv[5]);
//		m= HMHH(window_len);
//		std::cout << "window length" << window_len << std::endl;
//	}
//	else
//		m = HMHH();
//
	/*
	path = "C:\\Users\\adonis\\Desktop\\DataMiracle\\data\\gh\\sav\\";
	ofilename = "gh_HMHH.txt";
	start = 0;
	end = 1123;*/
	

//	std::ofstream ofile;
//	ofile.open(ofilename, std::ios::app);
//	
//	for (int index = start; index <= end; index++) {
//		char c[10];
//		_itoa(index, c, 10);
//		string t_path = path + c + ".png";
//		//std::cout << t_path << std::endl;
//		Mat src = imread(t_path, -1);
//		if (src.empty()) {
//			std::cout << "could not load image...\n" << t_path << std::endl;
//			continue;
//		}
//		/*float* feature = m.get_feature(src);
//		ofile << index-start << " ";
//		for (int i = 0; i < m.get_feature_len()-1; i++) {
//			ofile << feature[i]<<",";
//		}
//		ofile << feature[m.get_feature_len() - 1] << std::endl;*/
//		int result =  m.get_result(src);
//		imshow("Source", src);
//
//		std::cout << result << std::endl;
//	}
//
//	
//	
//	ofile.close();
//	return 0;
//}