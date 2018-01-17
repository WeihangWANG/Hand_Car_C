#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <zmq.h>
//#include "test.h"
#include "hand_pre.h"
#include "HMHH.h"
#include "HON4D.h"
#include <time.h>
#include "dmcam.h"
using namespace std;
using namespace cv;

//False实时,True仿真模式
bool SIM = FALSE;//FALSE;		
//void hand_seg(string filename) 
//{
//	//Mat src_u16 = imread("save.png", -1);
//	std::cout << filename << endl;
//	Mat src_u16 = imread(filename, -1);
//	//namedWindow("test opencv", CV_WINDOW_AUTOSIZE);
//	printf("rows=%d\n",src_u16.rows);
//	Mat src_u8(240, 320, CV_8UC1, Scalar(255));
//	Mat img(240, 320, CV_8UC1, Scalar(255));
//	Mat bin_img(240, 320, CV_8UC1, Scalar(255));
//	Mat mask, mask_img, rgb_img, roi_img;
//	src_u16.convertTo(src_u8, CV_8UC1);		//图像位数转换
//
//	//flip(src_u8, src_u8, 0);
//	//中值滤波
//	medianBlur(src_u8, img, 3);
//
//	//建立一维数组
//	int arr_img[76800];
//	int arr[76800];
//	int ind = 0;
//	for (int i = 0; i < img.rows; i++)
//	{
//		//得到第i行的首地址
//		uchar* data = img.ptr<uchar>(i);
//		for (int j = 0; j < img.cols; j++)
//		{
//			if (data[j] < 30)
//				data[j] = 255;
//			arr_img[ind] = data[j];
//			ind = ind + 1;
//		}
//	}
//	//升序排列像素值数组
//	sort(arr_img, arr_img+76800);
//	//取前4000个点计算平均深度阈值
//	int sum = 0;
//	for (int i = 0; i < 4000; i++)
//		sum = sum + arr_img[i];
//	int aver = sum / 3200;
//	if (aver < 115)
//		aver = 115;
//	else
//		aver = aver + 10;
//	std::cout << "aver=" << aver << endl;
//	////深度切割滤波
//	//for (int i = 0; i < img.rows; i++)
//	//{
//	//	//得到第i行的首地址
//	//	uchar* data = img.ptr<uchar>(i);
//	//	{
//	//	for (int j = 0; j < img.cols; j++)
//	//		if (data[j]>aver)
//	//			data[j] = 255;
//	//	}
//	//}
//	//图像二值化
//	medianBlur(img, img, 3);
//	threshold(img, bin_img, aver, 255, THRESH_BINARY);
//	bitwise_not(bin_img, bin_img);
//	//提取轮廓、连通域检测及绘制
//	vector<vector<Point> > contours;
//	vector<Point> max_cont;
//	findContours(bin_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	
//	double max_area = 0;
//	int n = 0;
//	for (int i = 0; i < int(contours.size()); i++)
//	{
//		if(contourArea(contours[i]) > max_area)
//		{
//			//cout << "if" << endl;
//			max_area = contourArea(contours[i]);
//			max_cont = contours[i];
//			if(i > 0)
//				drawContours(bin_img, contours, n, 0, -1);
//			n = i;
//		}
//		else
//		{
//			//cout << "else" << endl;
//			drawContours(bin_img, contours, i, 0, -1);
//		}
//	}
//	if (max_area > 8000)
//		drawContours(bin_img, contours, -1, 0, -1);
//	//腐蚀与膨胀
//	erode(bin_img, bin_img, Mat());
//	dilate(bin_img, bin_img, Mat());
//	//计算最小外接矩形
//	//vector<Rect> box;
//	Rect box = boundingRect(Mat(max_cont));
//	if (box.width > box.height)
//		rectangle(bin_img, Point(box.x+max(box.height,60), box.y),Point(320,240), Scalar(0), -1);
//	else
//		rectangle(bin_img, Point(box.x, box.y+max(60, box.width)), Point(320, 240), Scalar(0), -1);
//	//计算mask，进行手部提取
//	bin_img.convertTo(mask, CV_8UC1, 1.0 / 255);
//	mask_img = img.mul(mask);
//	medianBlur(mask_img, mask_img, 3);
//	int roi_side = max(60, min(box.width, box.height));
//	roi_img = mask_img(Rect(box.x, box.y, roi_side, roi_side));
//	resize(roi_img, roi_img, Size(80,80));
//	//计算最终轮廓，在原图上红色标注
//	vector<vector<Point>> fin_cont;
//	findContours(bin_img, fin_cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	cvtColor(img, rgb_img, COLOR_GRAY2BGR);
//	drawContours(rgb_img, fin_cont, -1, Scalar(0,0,255), -1);
//	cv::imshow("src", src_u16);     //显示原始16位图像
//	cv::imshow("src_u8", img);   //显示8位深度图
//	cv::imshow("bin", bin_img);		//显示二值化图像
//	//imshow("mask", mask_img);		//显示原深度图mask
//	cv::imshow("rgb", rgb_img);		//显示手部区域绿色填充
//	cv::imshow("roi", roi_img);		//显示手部切割ROI数据集
//	imwrite("mask.png", mask);
//	waitKey(200);
//}

int main()
{
	double dur;
	clock_t start, end;		 //计算运行时间
	std::cout << "Hand Recognition in Car Platform..." << endl;

	ofstream outfile;
	outfile.open("feature_HON4D.txt");		//写入文件，若没有则新建

	/*****************用于调试*******************/
	/*
	Mat test = cv::imread("D:\\Documents\\Downloads\\data0110\\data0110\\nph\\src\\221.png",0);
	cv::imshow("src", test);
	Mat test_mask = hand_seg(&test);
	cv::imshow("test",test_mask);
	cv::waitKey(0);
	*/

	//仿真模式
	if (SIM > 0)
	{
		std::cout << "sim mode NOW -_-";
		string ImgName;//图片名(绝对路径)
		ifstream finPos("infofile.txt");//正样本图片的文件名列表

		//类的实例化
		//Hand *hand = new Hand();
		/*HMHH hand = HMHH();
		HON4D *cHON4D;
		cHON4D = new HON4D(40, 40, 320, 240, 30, "projector\\");*/
		//const int len = hand.get_feature_len();
		//const int len1 = cHON4D->get_feature_len();
		int cnt = 2424;
		for (int num = 0 ; num < 694 ; num++)
		{
			stringstream id;
			id << num;
			//string ImgName = "D:\\Documents\\PyCharm\\Hand_in_CAR\\data\\wwh\\src\\" + id.str() + ".png";
			string ImgName = "D:\\Documents\\Downloads\\data0110\\data0110\\whh\\src\\" + id.str() + ".png";
			cout << ImgName << endl;
			Mat src_img = cv::imread(ImgName, -1);		//读取图片
//			cv::imshow("src", src_img);
			Mat roi_img = hand_seg(&src_img);		//手部分割
			//图片保存文件名int2str
			stringstream ss;
			ss << cnt;
			string name = "./dataset/" + ss.str() + ".png";
			cv::imwrite(name, roi_img);
			cnt = cnt + 1;
			cv::waitKey(10);
		}
		//for (int num = 0; num < 611 && getline(finPos, ImgName) ; num++)
		//{
		//	//cout << "处理：" << ImgName << endl;
		//	ImgName = "D:\\Documents\\Downloads\\data0110\\data0110\\nph\\src\\" + ImgName;			//样本的路径名
		//	cout << ImgName << endl;
		//	Mat src_img = cv::imread(ImgName,0);		//读取图片
		//	cv::imshow("src", src_img);
		//	Mat roi_img = hand_seg(&src_img);		//手部分割
		//	//float* feature;
		//	//feature = cHON4D->get_feature(roi_img);
		//	////保存特征向量
		//	//outfile << cnt << " ";
		//	//for (int i = 0; i < len1; i++)
		//	//	if (i < len1 - 1)
		//	//		outfile << feature[i] << ",";
		//	//	else
		//	//		outfile << feature[i] << endl;
		//	//图片保存文件名int2str
		//	stringstream ss;
		//	ss << cnt;
		//	string name = "./sav/" + ss.str() + ".png";
		//	//cout << name << endl;			//打印文件名
		//	cv::imwrite(name, roi_img);
		//	//Mat img_u8;
		//	//img_u16.convertTo(img_u8, CV_8UC1);
		//	//cv::imshow("src", img_u8);     //显示8位图像
		//	cv::waitKey(10);
		//	cnt = cnt + 1;
		//	
		//	//delete []feature;     //释放内存
		//}
	}
	//实时模式
	else
	{
		cout << "rt0" << endl;
		void* context = zmq_ctx_new();/// 创建一个新的环境  
		assert(context != NULL);

		int ret = zmq_ctx_set(context, ZMQ_MAX_SOCKETS, 1);/// 该环境中只允许有一个socket的存在  
		cout << "rt1" << endl;//cout << "2" << endl;
		assert(ret == 0);

		void* subscriber = zmq_socket(context, ZMQ_SUB);/// 创建一个订阅者  
		assert(subscriber != NULL);
		cout << "rt2" << endl;
		//ret = zmq_connect(subscriber, "tcp://192.168.7.2:56789");/// 连接到服务器  
		ret = zmq_connect(subscriber, "tcp://127.0.0.1:50660");/// 连接到服务器 
		assert(ret == 0);
		cout << "rt3" << endl;
		ret = zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "DIST", 0);/// 必须添加该语句对消息滤波，否则接受不到消息  
		assert(ret == 0);
		cout << "rt4" << endl;
		uchar buf[153612];/// 消息缓冲区 
		uint16_t data[320 * 240];
		int cnt = 0;
		cout << "rt5" << endl;

		uint16_t dst[320][240];
		cout << zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT) << endl;
		while (zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT))
		{
			std::cout << zmq_recv(subscriber, buf, 153612, ZMQ_DONTWAIT) << endl;
			//ret = /// 接收消息，非堵塞式
			//if (ret != -1)/// 打印消息  
			//{
				uint16_t *ptr = (uint16_t *)(&buf[12]);		//去掉帧头的原始数据
				for (int i = 0; i<320 * 240; i++)
				{
					data[i] = (float)ptr[i];
					//cout << data[i];
				}
				for (int i = 0; i < 320; i++)	//column
				{
					for (int j = 0; j < 240; j++)	//row
					{
						dst[i][j] = data[320 * j + i];
						//std::cout << dst[j][i] << ',';
					}
					//std::cout << endl;
				}
				Mat img_u16;
				img_u16 = Mat(320, 240, CV_16UC1, &dst);
				cv::transpose(img_u16, img_u16);
				cv::flip(img_u16, img_u16, 1);		//完成旋转90度
				cv::flip(img_u16, img_u16, 1);		//0：上下反转；1：水平翻转；-1：上下水平同时翻转
				Mat img_src = img_u16.clone();
				(img_src).convertTo(img_src, CV_8UC1);		//图像位数转换
				start = clock();
				Mat roi_img = hand_seg(&img_src);
				end = clock();
				std::cout << "preprocess time = " << (double)(end - start)/ CLOCKS_PER_SEC << endl;
				start = end;
				//特征向量的计算
				//HMHH feature
				//float* fea = hand.get_feature(roi_img);
				//outfile << cnt << " ";
				//for (int i = 0; i < len; i++)
				//	if (i < len - 1)
				//		outfile << fea[i] << ",";
				//	else
				//		outfile << fea[i] << endl;
				//end = clock();
				//int res = hand.get_result(roi_img);
				//char cls[2];
				//_itoa(res, cls, 10);
				//Mat clr_img;
				//cv::applyColorMap(img_src, clr_img, COLORMAP_JET);
				//cv::putText(clr_img, cls, Point(50, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);//在图片上写文字
				//cv::imshow("clr",clr_img);
				//std::cout << "frame result=" << res << endl;
				//std::cout << "HMHH time = " << (double)(end - start) / CLOCKS_PER_SEC << endl;
				//start = end;
				//////HON4D feature
				//float* feature;
				//feature = cHON4D->get_feature(roi_img);
				////保存特征向量
				//outfile << cnt << " ";
				//for (int i = 0; i < len1; i++)
				//	if (i < len1 - 1)
				//		outfile << feature[i] << ",";
				//	else
				//		outfile << feature[i] << endl;
				//end = clock();
				//cout << "HON4D time = " << (double)(end - start) / CLOCKS_PER_SEC << endl;
				//start = end;
				//图片保存文件名int2str
				stringstream ss;
				ss << cnt;
				string name = "./sav/"+ ss.str() + ".png";
				string src_name = "./src/" + ss.str() + ".png";
				//cout << name << endl;			//打印文件名
				//保存原始图与手部分割结果图片
				//cv::imwrite(name, roi_img);
				//cv::imwrite(src_name, img_src);
				Mat img_u8;
				img_u16.convertTo(img_u8, CV_8UC1);
				cv::imshow("src", img_u8 * 20);     //显示8位图像
				cv::waitKey(100);
				cnt = cnt + 1;
				//delete []fea;
				//delete []feature;     //释放内存
			//}
			//std::cout << "finish single frame" << endl;
				memset(buf,0,sizeof(uchar)*153612);
				Sleep(1);
		}
	}	
	outfile.close();
}