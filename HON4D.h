#ifndef HON4D_H_
#define HON4D_H_
//#include <cv.h>
#include <opencv2\opencv.hpp>
//#include <highgui.h>
#include <string>
#include "svm.h" 
using namespace std;

#define PI 3.1415926
#define binThreshold 1.3090

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

inline void SafeImgDel(IplImage * imgRelease)
{
	if (imgRelease != NULL) cvReleaseImage(&imgRelease);
	imgRelease = NULL;
}

inline void SafeMatDel(CvMat * matRelease)
{
	if (matRelease != NULL) cvReleaseMat(&matRelease);
	matRelease = NULL;
}

typedef struct _tHON4DCell{
	int width;	//width used to build cell,default 8
	int height;	//height used to build cell,default 8
	int depth;
	int numR;	//cell number of each row
	int numC;	//cell number of each column
	int numD;	//cell number of each column
	int	bin_n_HON4D;//Bin number of theta in HON4D
}HON4D_cell;

typedef struct _tHON4DWindow{
	int width;	//width used for detection window, default 64
	int height;	//height used for detection window, default 128
	int	depth;//overall feature length for detection window
	int	featLen;//overall feature length for detection window
}HON4D_win;

typedef struct tfeat{
	float* feature;
	int count;
};

class HON4D{

public:
	HON4D_cell cell;
	HON4D_win DetWin;
	HON4D(int nCellWidth, int nCellHeight, int nWinWidth, int nWinHeight, int nWinDepth, string vidPath = "H:\\Code\\HON4D\\HON4D\\Data\\Projectors\\");
	~HON4D();
	//void find_png(const char *dir_path, const char *suffix, vector<string>& files);
	float* get_feature(cv::Mat cur);
	int get_result(cv::Mat cur);
	int load_para(string modelPath, string max_path, string min_path);
	int get_feature_len();
	
private:
	CvMat* p;
	cv::Mat last;
	vector<tfeat> q;
	int cnt = 0;
	bool chushi = 1;
	svm_model *svmModel;
	float* feature_max;
	float* feature_min;
	void compute_gradient(CvMat* cur, CvMat* last, CvMat *dxLine, CvMat *dyLine, CvMat *dzLine);
	void compute_HON4D_hist(int dx, int dy, int dz, float* hist);
	void compute_single_hist(CvMat* dxLine, CvMat* dyLine, CvMat* dzLine);
	int normalize_feature(float *feature_vector);
	void hist_statistic(vector<tfeat> q, float* final_feat);
	int readProjectors(int pNo, string projectorsPath, CvMat* &p);
	void scale(float* feature, float* feature_max, float* feature_min);
	//std::wstring s2ws(const std::string& s);
};

#endif /* HON4D_H_ */
