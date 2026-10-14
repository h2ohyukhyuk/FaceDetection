#pragma once
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#define CV_RGB(r,g,b) cvScalar((b),(g),(r),0)
struct CvTypeInfo { const char* type_name; };
inline CvTypeInfo* cvTypeOf(const void* x) {
 static CvTypeInfo image={CV_TYPE_NAME_IMAGE}, mat={CV_TYPE_NAME_MAT};
 return CV_IS_IMAGE(x) ? &image : &mat;
}
