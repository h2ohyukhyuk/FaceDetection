#pragma once
#define CV_LOAD_IMAGE_COLOR 1
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core_c.h>
inline IplImage* cvLoadImage(const char* path, int flags=1) {
    cv::Mat m=cv::imread(path, flags);
    if(m.empty()) return nullptr;
    IplImage header=cvIplImage(m);
    return cvCloneImage(&header);
}
