#define _CRT_SECURE_NO_WARNINGS
#include "HON4D.h"
#include <iostream>
#include <vector>
#include <conio.h>
#include <fstream>
#include <exception>
#include <windows.h>
#include <time.h>
#include <tchar.h>
#include <cstdlib>
#include <string>

using namespace std;

void HON4D::compute_gradient(CvMat* cur, CvMat* last, CvMat *dxLine, CvMat *dyLine, CvMat *dzLine)
{
	int dx, dy, dz;
	CvScalar left, right, up, down, current, next;

	int nImg_w = cur->width;
	int nImg_h = cur->height;
	int PixBegin = 0;
	int HON4D_difference = 5;
	int bin_n_HON4D = cell.bin_n_HON4D;
	float *temp_hist = new float[bin_n_HON4D];

	for (int i = PixBegin; i<nImg_h - PixBegin; i++)
	{
		for (int j = PixBegin; j < nImg_w - PixBegin; j++)
		{
			if (i < HON4D_difference || i >= nImg_h - HON4D_difference || j < HON4D_difference || j >= nImg_w - HON4D_difference)
			{
				cvSet2D(dxLine, i, j, cvScalar(0));
				cvSet2D(dyLine, i, j, cvScalar(0));
				cvSet2D(dzLine, i, j, cvScalar(0));
				continue;
			}

			//get 4 pixel values around this pixel
			left = cvGet2D(last, i, j - HON4D_difference);
			right = cvGet2D(last, i, j + HON4D_difference);
			up = cvGet2D(last, i - HON4D_difference, j);
			down = cvGet2D(last, i + HON4D_difference, j);
			current = cvGet2D(last, i, j);
			next = cvGet2D(cur, i, j);

			//compute dx dy
			dx = int(right.val[0] - left.val[0]);
			dy = int(down.val[0] - up.val[0]);
			dz = int(current.val[0] - next.val[0]);

			cvSet2D(dxLine, i, j, cvScalar(dx));
			cvSet2D(dyLine, i, j, cvScalar(dy));
			cvSet2D(dzLine, i, j, cvScalar(dz));
		}
	}
}

void HON4D::compute_HON4D_hist(int dx, int dy, int dz, float* hist)
{
	float fi, fj, fk, f_value = 0.0;
	//int which_bin;

	fi = float(dx);
	fj = float(dy);
	fk = float(dz);
	float gmag = sqrt(fi*fi + fj*fj + fk*fk);

	CvScalar s;
	CvMat* d;
	CvMat* mulres;
	CvMat* magMat;

	d = cvCreateMat(4, 1, CV_32FC1);

	cvSet2D(d, 0, 0, cvScalar(fi));
	cvSet2D(d, 1, 0, cvScalar(fj));
	cvSet2D(d, 2, 0, cvScalar(fk));
	cvSet2D(d, 3, 0, cvScalar(-1));

	mulres = cvCreateMat(cell.bin_n_HON4D, 1, CV_32FC1);
	cvMatMul(p, d, mulres);

	// prepare gmag matrix
	magMat = cvCreateMat(cell.bin_n_HON4D, 1, CV_32FC1);
	cvSetZero(magMat);
	cvAddS(magMat, cvScalar(gmag), magMat);

	//divide result over gmag
	cvDiv(mulres, magMat, mulres);

	// subtract threshold
	cvSubS(mulres, cvScalar(binThreshold), mulres);

	// remove all negative values
	for (int i = 0; i<cell.bin_n_HON4D; i++) {
		s = cvGet2D(mulres, i, 0);

		if (s.val[0]<0){
			cvSet2D(mulres, i, 0, cvScalar(0));
		}
	}

	// compute vec magnitude
	float vecmag = 0;
	for (int i = 0; i<cell.bin_n_HON4D; i++) {
		s = cvGet2D(mulres, i, 0);
		vecmag = vecmag + (s.val[0] * s.val[0]);
	}
	vecmag = sqrt(vecmag);

	if (vecmag != 0){
		//prepare vecmagmat 
		cvSetZero(magMat);
		cvAddS(magMat, cvScalar(vecmag), magMat);

		//divide result over vecmag
		cvDiv(mulres, magMat, mulres);

		// copy result to histogram
		for (int i = 0; i<cell.bin_n_HON4D; i++) {
			s = cvGet2D(mulres, i, 0);
			hist[i] = hist[i] + s.val[0];
		}
	}
	SafeMatDel(mulres);
	SafeMatDel(magMat);
	SafeMatDel(d);
}

void HON4D::compute_single_hist(CvMat* dxLine, CvMat* dyLine, CvMat* dzLine)
{
	int nPointx = 0;
	int nPointy = 0;
	int bin_n_HON4D = cell.bin_n_HON4D;
	float *temp_hist = new float[bin_n_HON4D];

	int image_w = dxLine->width;
	int image_h = dxLine->height;

	int win_w = DetWin.width;
	int win_h = DetWin.height;

	int win_end_point_x = nPointx + win_w - 1;
	int win_end_point_y = nPointy + win_h - 1;

	if (win_end_point_x>image_w || win_end_point_y>image_h)
	{
		printf("HON4D error! Detection window gets out of image!!!\n");
		exit(0);
	}

	int ci, cj, xmin, ymin, xmax, ymax, x, y, dx, dy, dz;

	tfeat pFeatOut;
	pFeatOut.feature = new float[cell.numC*cell.numR*cell.bin_n_HON4D];
	pFeatOut.count = 0;
	float *pFeatOut_pointer = pFeatOut.feature;

	for (ci = 0; ci<cell.numR; ci++)
	{
		for (cj = 0; cj<cell.numC; cj++)
		{
			memset(temp_hist, 0.0f, bin_n_HON4D*sizeof(float));

			xmin = nPointx + cell.width * ci;
			ymin = nPointy + cell.height * cj;
			xmax = xmin + cell.width - 1;
			ymax = ymin + cell.height - 1;

			//for each pixel in this cell		
			for (x = xmin; x <= xmax; x++)
			{
				for (y = ymin; y <= ymax; y++)
				{
					//find out which bin to vote
					dx = cvGet2D(dxLine, y, x).val[0];
					dy = cvGet2D(dyLine, y, x).val[0];
					dz = cvGet2D(dzLine, y, x).val[0];

					if (dx == 0 || dy == 0 || dz == 0)
						continue;

					compute_HON4D_hist(dx, dy, dz, temp_hist);

				}
			}

			pFeatOut.count++;
			//concatenate histogram in cells
			memcpy(pFeatOut_pointer, temp_hist, bin_n_HON4D*sizeof(float));

			// if not last iteration
			pFeatOut_pointer = pFeatOut_pointer + bin_n_HON4D;
		}
	}

	q.insert(q.begin() + cnt, pFeatOut);
	
	if (cnt < DetWin.depth - 1)
	{
		cnt += 1;
	}
	else
	{
		cnt = cnt;
		q.erase(q.begin());
	}

	delete[] temp_hist;
	pFeatOut_pointer = NULL;
}

int HON4D::normalize_feature(float *feature_vector)
{
	int feature_dim = cell.bin_n_HON4D;

	float total_n = 0;

	for (int indHist = 0; indHist<feature_dim; indHist++){
		total_n = total_n + feature_vector[indHist];
	}

	int i;

	if (total_n>0)
	for (i = 0; i < feature_dim; i++)
	{
		feature_vector[i] = feature_vector[i] / total_n;
		//cout << feature_vector[i] << " ";
	}

	return 1;
}

void HON4D::hist_statistic(vector<tfeat> q, float* final_feat)
{
	int length = q.size();
	int seg1 = length / 3;
	int seg2 = seg1 * 2;

	if (length == DetWin.depth - 1)
	{
		float* final_f = new float[q[0].count*cell.bin_n_HON4D];
		float* final_s = new float[q[0].count*cell.bin_n_HON4D];
		float* final_t = new float[q[0].count*cell.bin_n_HON4D];
		memset(final_f, 0.0f, q[0].count*cell.bin_n_HON4D*sizeof(float));
		memset(final_s, 0.0f, q[0].count*cell.bin_n_HON4D*sizeof(float));
		memset(final_t, 0.0f, q[0].count*cell.bin_n_HON4D*sizeof(float));

		for (int i = 0; i < seg1; i++)
		{
			for (int j = 0; j < q[0].count*cell.bin_n_HON4D; j++)
			{
				final_f[j] = final_f[j] + q[i].feature[j];
				final_s[j] = final_s[j] + q[i + seg1].feature[j];
				final_t[j] = final_t[j] + q[i + seg2].feature[j];
			}
		}

		for (int k = 0; k < q[0].count; k++)
		{
			int bin_size = cell.bin_n_HON4D;

			float *temp = new float[bin_size];
			float* t_1 = final_f;
			memcpy(temp, t_1 + k*bin_size, bin_size*sizeof(float));
			normalize_feature(temp);
			memcpy(t_1 + k*bin_size, temp, bin_size*sizeof(float));

			float *temp_s = new float[bin_size];
			float* t_2 = final_s;
			memcpy(temp_s, t_2 + k*bin_size, bin_size*sizeof(float));
			normalize_feature(temp_s);
			memcpy(t_2 + k*bin_size, temp_s, bin_size*sizeof(float));

			float *temp_t = new float[bin_size];
			float* t_3 = final_t;
			memcpy(temp_t, t_3 + k*bin_size, bin_size*sizeof(float));
			normalize_feature(temp_t);
			memcpy(t_3 + k*bin_size, temp_t, bin_size*sizeof(float));

			t_1 = NULL;
			t_2 = NULL;
			t_3 = NULL;
		}

		float* t_f = final_feat;
		memcpy(t_f, final_f, cell.numC*cell.numR*cell.bin_n_HON4D*sizeof(float));
		t_f += cell.numC*cell.numR*cell.bin_n_HON4D;
		memcpy(t_f, final_s, cell.numC*cell.numR*cell.bin_n_HON4D*sizeof(float));
		t_f += cell.numC*cell.numR*cell.bin_n_HON4D;
		memcpy(t_f, final_t, cell.numC*cell.numR*cell.bin_n_HON4D*sizeof(float));
		t_f = NULL;

		delete[] final_f, final_s, final_t;
	}
	else
	{
		for (int i = 0; i < DetWin.featLen; i++)
		{
			final_feat[i] = 0;
		}
	}
}

int HON4D::readProjectors(int pNo, string projectorsPath, CvMat* &p)
{
	ifstream pfile;
	char buffer[4];
	sprintf(buffer, "%03d", pNo);
	string projFile = projectorsPath + buffer + ".txt";
	pfile.open(projFile);
	string line;
	int ind = 0;
	float v;

	getline(pfile, line);
	int sz = atof(line.c_str());
	p = cvCreateMat(sz, 4, CV_32FC1);

	while (pfile.good())
	{
		getline(pfile, line);
		if (line.length()<2)
			continue;
		//cout << line << endl;


		stringstream   linestream(line);
		string         value;

		getline(linestream, value, ',');
		v = atof(value.c_str());
		cvSet2D(p, ind, 0, cvScalar(v));

		getline(linestream, value, ',');
		v = atof(value.c_str());
		cvSet2D(p, ind, 1, cvScalar(v));


		getline(linestream, value, ',');
		v = atof(value.c_str());
		cvSet2D(p, ind, 2, cvScalar(v));

		getline(linestream, value, ',');
		v = atof(value.c_str());
		cvSet2D(p, ind, 3, cvScalar(v));


		ind++;
	}
	pfile.close();
	return sz;
}

HON4D::HON4D(int nCellWidth = 16, int nCellHeight = 16, int nWinWidth = 320, int nWinHeight = 240, int nWinDepth = 30, string vidPath)
{
	cell.width = nCellWidth;
	cell.height = nCellHeight;

	DetWin.width = nWinWidth;
	DetWin.height = nWinHeight;
	DetWin.depth = nWinDepth;

	cell.depth = DetWin.depth / 3;

	cell.numR = DetWin.width / cell.width;
	cell.numC = DetWin.height / cell.height;
	cell.numD = 3;

	int pNo = 0;
	//string projectorsPath = "H:\\Code\\HON4D\\HON4D\\Data\\Projectors\\";
	int projDim = readProjectors(pNo, vidPath, p);

	cell.bin_n_HON4D = projDim;
	DetWin.featLen = cell.numD * cell.numR * cell.numC * cell.bin_n_HON4D;

	/*
	tfeat ini;
	ini.feature = new float[cell.numC*cell.numR*cell.bin_n_HON4D];
	memset(ini.feature, 0.0f, cell.numR*cell.numC*cell.bin_n_HON4D*sizeof(float));
	ini.count = 0;

	for (int i = 0; i < DetWin.depth; i++)
	{
		q.insert(q.begin() + i, ini);
	}
	*/
}

HON4D::~HON4D()
{

}

int HON4D::get_feature_len()
{
	return DetWin.featLen;
}

float* HON4D::get_feature(cv::Mat  u_src)
{
	cv::Mat cur = cv::Mat(cv::Size(120, 120), CV_8UC1);
	cv::resize(cur, u_src, cv::Size(120, 120));
	float* final_feat=new float[DetWin.featLen];
	if (chushi==FALSE)
	{
		CvMat temp = cur;
		CvMat temp_last = last;
		CvMat *dxLine = cvCreateMat(cur.rows, cur.cols, CV_32FC1);
		CvMat *dyLine = cvCreateMat(cur.rows, cur.cols, CV_32FC1);
		CvMat *dzLine = cvCreateMat(cur.rows, cur.cols, CV_32FC1);
		compute_gradient(&temp, &temp_last, dxLine, dyLine, dzLine);
		compute_single_hist(dxLine, dyLine, dzLine);
		hist_statistic(q, final_feat);
		last = cur.clone();
	}
	else
	{
		last = cur.clone();
		chushi = FALSE;
		for (int i = 0; i < DetWin.featLen; i++)
		{
			final_feat[i] = 0;
		}
	}
	return final_feat;
}

void HON4D::scale(float* feature, float* feature_max, float* feature_min)
{
	int lower = -1;
	int upper = 1;

	/* pass 2: scale */
	for (int j = 0; j < DetWin.featLen; j++)
	{
		feature[j] = lower + (upper - lower) *
			(feature[j] - feature_min[j]) /
			(feature_max[j] - feature_min[j]);
	}

}

int HON4D::load_para(string modelPath, string max_path, string min_path)
{
	// 读取模型
	svm_model *svmModel = svm_load_model(modelPath.c_str());

	ifstream max_save, min_save;
	max_save.open(max_path.c_str());
	min_save.open(min_path.c_str());
	while (!max_save.eof())
	{
		max_save >> *feature_max;
		feature_max++;
	}
	while (!min_save.eof())
	{
		min_save >> *feature_min;
		feature_min++;
	}
	max_save.close();
	min_save.close();
	return 1;
}

int HON4D::get_result(cv::Mat cur)
{
	float* feat = new float[DetWin.featLen];
	feat = get_feature(cur);

	// 归一化
	scale(feat, feature_max, feature_min);

	svm_node *x = new svm_node[DetWin.featLen + 1];
	for (int i = 0; i < DetWin.featLen; i++)
	{
		if (feat[i] != 0)
		{
			x[i].index = i + 1;
			x[i].value = feat[i];
		}		
	}
	x[DetWin.featLen].index = -1;

	// 预测
	int predictValue = svm_predict(svmModel, x);
	return predictValue;
}