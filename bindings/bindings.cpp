////////////////////////////////////////////////////////////////////////////////
// AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015, University of California, Irvine
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the UC Irvine.
// 4. Neither the name of the UC Irvine nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

#include "hunter.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"

#include <emscripten/bind.h>

using namespace emscripten;
using namespace cv;
using namespace cv::flann;
using namespace cv::ml;

namespace Utils{

    template<typename T>
    emscripten::val data(const cv::Mat& mat) {
        return emscripten::val(emscripten::memory_view<T>( (mat.total()*mat.elemSize())/sizeof(T), (T*) mat.data));
    }

    emscripten::val matPtrI(const cv::Mat& mat, int i) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(0), mat.ptr(i)));
    }

    emscripten::val matPtrII(const cv::Mat& mat, int i, int j) {
        return emscripten::val(emscripten::memory_view<uint8_t>(mat.step1(1), mat.ptr(i,j)));
    }

    emscripten::val  matFromArray(const emscripten::val& object, int type) {
        int w=  object["width"].as<unsigned>();
        int h=  object["height"].as<unsigned>();
        std::string str = object["data"]["buffer"].as<std::string>();
        
        return emscripten::val(cv::Mat(h, w, type, (void*)str.data(), 0));
    }

    cv::Mat* createMat(Size size, int type, intptr_t data, size_t step) {
        return new cv::Mat(size, type, reinterpret_cast<void*>(data), step);
    }


    cv::Mat* createMat2(const std::vector<unsigned char>& vector) {
        return new cv::Mat(vector, false);
    }

    // returning MatSize
    static std::vector<int> getMatSize(const cv::Mat& mat)
    {
      std::vector<int> size;
      for (int i = 0; i < mat.dims; i++) {
        size.push_back(mat.size[i]);
      }
      return size;
    }

    static Mat eye(int rows, int cols, int type) {
      return Mat(cv::Mat::eye(rows, cols, type));
    }

    static Mat eye(Size size, int type) {
      return Mat(cv::Mat::eye(size, type));
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta) {
        obj.convertTo(m, rtype, alpha, beta);
    }

    Size matSize(const cv::Mat& mat) {
        return  mat.size();
    }
    cv::Mat mat_zeros_iii(int arg0, int arg1, int arg2) {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }
    cv::Mat mat_zeros_Si(cv::Size arg0, int arg1) {
        return cv::Mat::zeros(arg0,arg1);
    }
    cv::Mat mat_zeros_ipii(int arg0, const int* arg1, int arg2) {
        return cv::Mat::zeros(arg0,arg1,arg2);
    }
    cv::Mat mat_ones_iii(int arg0, int arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat mat_ones_ipii(int arg0, const int* arg1, int arg2) {
        return cv::Mat::ones(arg0, arg1, arg2);
    }
    cv::Mat mat_ones_Si(cv::Size arg0, int arg1) {
        return cv::Mat::ones(arg0, arg1);
    }

    double matDot(const cv::Mat& obj, const Mat& mat) {
        return  obj.dot(mat);
    }
    Mat matMul(const cv::Mat& obj, const Mat& mat, double scale) {
        return  Mat(obj.mul(mat, scale));
    }
    Mat matT(const cv::Mat& obj) {
        return  Mat(obj.t());
    }
    Mat matInv(const cv::Mat& obj, int type) {
        return  Mat(obj.inv(type));
    }

}

EMSCRIPTEN_BINDINGS(Utils) {

    register_vector<int>("IntVector");
    register_vector<char>("CharVector");
    register_vector<unsigned>("UnsignedVector");
    register_vector<unsigned char>("UCharVector");
    register_vector<std::string>("StrVector");
    register_vector<emscripten::val>("EmvalVector");
    register_vector<float>("FloatVector");
    register_vector<std::vector<int>>("IntVectorVector");
    register_vector<std::vector<Point>>("PointVectorVector");
    register_vector<cv::Point>("PointVector");
    register_vector<cv::Vec4i>("Vec4iVector");
    register_vector<cv::Mat>("MatVector");
    register_vector<cv::KeyPoint>("KeyPointVector");
    register_vector<cv::Rect>("RectVector");
    register_vector<std::vector<cv::Rect>>("RectVectorVector");
    register_vector<cv::Point2f>("Point2fVector");
    register_vector<cv::DMatch>("DMatchVector");
    register_vector<std::vector<cv::DMatch>>("DMatchVectorVector");
    register_vector<std::vector<char>>("CharVectorVector");

    register_vector<ph::PixelMatchingResult>("PixelMatchingResultVector");

    emscripten::class_<ph::PixelMatchingResult>("PixelMatchingResult")
        .constructor()
        .property("isMatched", &ph::PixelMatchingResult::isMatched)
        .property("center1", &ph::PixelMatchingResult::center1)
        .property("bounding1", &ph::PixelMatchingResult::bounding1)
        .property("center2", &ph::PixelMatchingResult::center2)
        .property("bounding2", &ph::PixelMatchingResult::bounding2)
        .property("diffMarkers1", &ph::PixelMatchingResult::diffMarkers1)
        .property("diffMarkers2", &ph::PixelMatchingResult::diffMarkers2)
        ;

    emscripten::class_<ph::DiffConfig>("DiffConfig")
        .constructor<>()
        .property("_debug", &ph::DiffConfig::debug)
        .property("maxMatchingPoints", &ph::DiffConfig::maxMatchingPoints)
        .property("connectionDistance", &ph::DiffConfig::connectionDistance)
        .property("thresholdPixcelNorm", &ph::DiffConfig::thresholdPixelNorm)
        .property("gridSize", &ph::DiffConfig::gridSize)
        ;

    emscripten::class_<ph::DiffResult>("DiffResult")
        .constructor<>()
        .property("matches", &ph::DiffResult::matches)
        .property("strayingRects1", &ph::DiffResult::strayingRects1)
        .property("strayingRects2", &ph::DiffResult::strayingRects2)
        ;

    function("_detectDiff", select_overload<void(const cv::Mat&, const cv::Mat&, ph::DiffResult&, const ph::DiffConfig&)>(&ph::detectDiff));

    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        //.constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor(&Utils::createMat, allow_raw_pointers())
        .constructor(&Utils::createMat2, allow_raw_pointers())
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        //.function("assignTo", select_overload<void(Mat&, int)const>(&cv::Mat::assignTo))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo",  select_overload<void(const Mat&, Mat&, int, double, double)>(&Utils::convertTo))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .class_function("eye",select_overload<Mat(int, int, int)>(&Utils::eye))
        .class_function("eye",select_overload<Mat(Size, int)>(&Utils::eye))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))

        .function("copyTo", select_overload<void(OutputArray)const>(&cv::Mat::copyTo))
        .function("copyTo", select_overload<void(OutputArray, InputArray)const>(&cv::Mat::copyTo))
        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))

        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .class_function("ones",select_overload<Mat(int, int, int)>(&Utils::mat_ones_iii))
        .class_function("ones",select_overload<Mat(Size, int)>(&Utils::mat_ones_Si))
        .class_function("zeros",select_overload<Mat(int, int, int)>(&Utils::mat_zeros_iii))
        .class_function("zeros",select_overload<Mat(Size, int)>(&Utils::mat_zeros_Si))
        .function("depth", select_overload<int()const>(&cv::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&cv::Mat::col))

        .function("dot", select_overload<double(const Mat&, const Mat&)>(&Utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&Utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&Utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&Utils::matT))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)

        .function("data", &Utils::data<unsigned char>)
        .function("data8S", &Utils::data<char>)
        .function("data16u", &Utils::data<unsigned short>)
        .function("data16s", &Utils::data<short>)
        .function("data32s", &Utils::data<int>)
        .function("data32f", &Utils::data<float>)
        .function("data64f", &Utils::data<double>)

        .function("ptr", select_overload<val(const Mat&, int)>(&Utils::matPtrI))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&Utils::matPtrII))

        .function("size" , &Utils::getMatSize)
        .function("get_uchar_at" , select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_uchar_at", select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("get_ushort_at", select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_ushort_at", select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("get_int_at" , select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("get_int_at", select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("get_double_at", select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("get_double_at", select_overload<double&(int, int)>(&cv::Mat::at<double>))
        .function("get_float_at", select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("get_float_at", select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function( "getROI_Rect", select_overload<Mat(const Rect&)const>(&cv::Mat::operator()));

    emscripten::class_<cv::Vec<int,4>>("Vec4i")
        .constructor<>()
        .constructor<int, int, int, int>();

    emscripten::class_<cv::RNG> ("RNG");

    value_array<Size>("Size")
        .element(&Size::height)
        .element(&Size::width);


    value_array<Point>("Point")
        .element(&Point::x)
        .element(&Point::y);

    value_array<Point2f>("Point2f")
        .element(&Point2f::x)
        .element(&Point2f::y);

    emscripten::class_<cv::Rect_<int>> ("Rect")
        .constructor<>()
        .constructor<const cv::Point_<int>&, const cv::Size_<int>&>()
        .constructor<int, int, int, int>()
        .constructor<const cv::Rect_<int>&>()
        .property("x", &cv::Rect_<int>::x)
        .property("y", &cv::Rect_<int>::y)
        .property("width", &cv::Rect_<int>::width)
        .property("height", &cv::Rect_<int>::height);

    emscripten::class_<cv::Scalar_<double>> ("Scalar")
        .constructor<>()
        .constructor<double>()
        .constructor<double, double>()
        .constructor<double, double, double>()
        .constructor<double, double, double, double>()
        .class_function("all", &cv::Scalar_<double>::all)
        .function("isReal", select_overload<bool()const>(&cv::Scalar_<double>::isReal));

    function("matFromArray", &Utils::matFromArray);

    constant("CV_8UC1", CV_8UC1) ;
    constant("CV_8UC2", CV_8UC2) ;
    constant("CV_8UC3", CV_8UC3) ;
    constant("CV_8UC4", CV_8UC4) ;

    constant("CV_8SC1", CV_8SC1) ;
    constant("CV_8SC2", CV_8SC2) ;
    constant("CV_8SC3", CV_8SC3) ;
    constant("CV_8SC4", CV_8SC4) ;

    constant("CV_16UC1", CV_16UC1) ;
    constant("CV_16UC2", CV_16UC2) ;
    constant("CV_16UC3", CV_16UC3) ;
    constant("CV_16UC4", CV_16UC4) ;

    constant("CV_16SC1", CV_16SC1) ;
    constant("CV_16SC2", CV_16SC2) ;
    constant("CV_16SC3", CV_16SC3) ;
    constant("CV_16SC4", CV_16SC4) ;

    constant("CV_32SC1", CV_32SC1) ;
    constant("CV_32SC2", CV_32SC2) ;
    constant("CV_32SC3", CV_32SC3) ;
    constant("CV_32SC4", CV_32SC4) ;

    constant("CV_32FC1", CV_32FC1) ;
    constant("CV_32FC2", CV_32FC2) ;
    constant("CV_32FC3", CV_32FC3) ;
    constant("CV_32FC4", CV_32FC4) ;

    constant("CV_64FC1", CV_64FC1) ;
    constant("CV_64FC2", CV_64FC2) ;
    constant("CV_64FC3", CV_64FC3) ;
    constant("CV_64FC4", CV_64FC4) ;

    constant("CV_8U", CV_8U);
    constant("CV_8S", CV_8S);
    constant("CV_16U", CV_16U);
    constant("CV_16S", CV_16S);
    constant("CV_32S",  CV_32S);
    constant("CV_32F", CV_32F);
    constant("CV_32F", CV_32F);


    constant("BORDER_CONSTANT", +cv::BorderTypes::BORDER_CONSTANT);
    constant("BORDER_REPLICATE", +cv::BorderTypes::BORDER_REPLICATE);
    constant("BORDER_REFLECT", +cv::BorderTypes::BORDER_REFLECT);
    constant("BORDER_WRAP", +cv::BorderTypes::BORDER_WRAP);
    constant("BORDER_REFLECT_101", +cv::BorderTypes::BORDER_REFLECT_101);
    constant("BORDER_TRANSPARENT", +cv::BorderTypes::BORDER_TRANSPARENT);
    constant("BORDER_REFLECT101", +cv::BorderTypes::BORDER_REFLECT101);
    constant("BORDER_DEFAULT", +cv::BorderTypes::BORDER_DEFAULT);
    constant("BORDER_ISOLATED", +cv::BorderTypes::BORDER_ISOLATED);

    constant("NORM_INF", +cv::NormTypes::NORM_INF);
    constant("NORM_L1", +cv::NormTypes::NORM_L1);
    constant("NORM_L2", +cv::NormTypes::NORM_L2);
    constant("NORM_L2SQR", +cv::NormTypes::NORM_L2SQR);
    constant("NORM_HAMMING", +cv::NormTypes::NORM_HAMMING);
    constant("NORM_HAMMING2", +cv::NormTypes::NORM_HAMMING2);
    constant("NORM_TYPE_MASK", +cv::NormTypes::NORM_TYPE_MASK);
    constant("NORM_RELATIVE", +cv::NormTypes::NORM_RELATIVE);
    constant("NORM_MINMAX", +cv::NormTypes::NORM_MINMAX);

    constant("INPAINT_NS", +cv::INPAINT_NS);
    constant("INPAINT_TELEA", +cv::INPAINT_TELEA);

}
namespace Wrappers {
    void Canny_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, bool arg6) {
        return cv::Canny(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void GaussianBlur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, double arg4, double arg5, int arg6) {
        return cv::GaussianBlur(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void HoughCircles_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, double arg6, double arg7, int arg8, int arg9) {
        return cv::HoughCircles(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void HoughLines_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, double arg6, double arg7, double arg8, double arg9) {
        return cv::HoughLines(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void HoughLinesP_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, double arg6, double arg7) {
        return cv::HoughLinesP(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void HuMoments_wrapper(const Moments& arg1, cv::Mat& arg2) {
        return cv::HuMoments(arg1, arg2);
    }
    
    void LUT_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::LUT(arg1, arg2, arg3);
    }
    
    void Laplacian_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, double arg5, double arg6, int arg7) {
        return cv::Laplacian(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    double Mahalanobis_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::Mahalanobis(arg1, arg2, arg3);
    }
    
    void PCABackProject_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::PCABackProject(arg1, arg2, arg3, arg4);
    }
    
    void PCACompute_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::PCACompute(arg1, arg2, arg3, arg4);
    }
    
    void PCACompute_wrapper1(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, double arg4) {
        return cv::PCACompute(arg1, arg2, arg3, arg4);
    }
    
    void PCAProject_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::PCAProject(arg1, arg2, arg3, arg4);
    }
    
    double PSNR_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::PSNR(arg1, arg2);
    }
    
    void SVBackSubst_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, cv::Mat& arg5) {
        return cv::SVBackSubst(arg1, arg2, arg3, arg4, arg5);
    }
    
    void SVDecomp_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5) {
        return cv::SVDecomp(arg1, arg2, arg3, arg4, arg5);
    }
    
    void Scharr_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, double arg6, double arg7, int arg8) {
        return cv::Scharr(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    void Sobel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, int arg6, double arg7, double arg8, int arg9) {
        return cv::Sobel(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void absdiff_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::absdiff(arg1, arg2, arg3);
    }
    
    void accumulate_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::accumulate(arg1, arg2, arg3);
    }
    
    void accumulateProduct_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::accumulateProduct(arg1, arg2, arg3, arg4);
    }
    
    void accumulateSquare_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::accumulateSquare(arg1, arg2, arg3);
    }
    
    void accumulateWeighted_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, const cv::Mat& arg4) {
        return cv::accumulateWeighted(arg1, arg2, arg3, arg4);
    }
    
    void adaptiveThreshold_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, int arg4, int arg5, int arg6, double arg7) {
        return cv::adaptiveThreshold(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void add_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4, int arg5) {
        return cv::add(arg1, arg2, arg3, arg4, arg5);
    }
    
    void addWeighted_wrapper(const cv::Mat& arg1, double arg2, const cv::Mat& arg3, double arg4, double arg5, cv::Mat& arg6, int arg7) {
        return cv::addWeighted(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void applyColorMap_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::applyColorMap(arg1, arg2, arg3);
    }
    
    void approxPolyDP_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, bool arg4) {
        return cv::approxPolyDP(arg1, arg2, arg3, arg4);
    }
    
    double arcLength_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::arcLength(arg1, arg2);
    }
    
    void arrowedLine_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7, double arg8) {
        return cv::arrowedLine(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    void batchDistance_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, cv::Mat& arg5, int arg6, int arg7, const cv::Mat& arg8, int arg9, bool arg10) {
        return cv::batchDistance(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }
    
    void bilateralFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, int arg6) {
        return cv::bilateralFilter(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void bitwise_and_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_and(arg1, arg2, arg3, arg4);
    }
    
    void bitwise_not_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::bitwise_not(arg1, arg2, arg3);
    }
    
    void bitwise_or_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_or(arg1, arg2, arg3, arg4);
    }
    
    void bitwise_xor_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::bitwise_xor(arg1, arg2, arg3, arg4);
    }
    
    void blur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, Point arg4, int arg5) {
        return cv::blur(arg1, arg2, arg3, arg4, arg5);
    }
    
    int borderInterpolate_wrapper(int arg1, int arg2, int arg3) {
        return cv::borderInterpolate(arg1, arg2, arg3);
    }
    
    Rect boundingRect_wrapper(const cv::Mat& arg1) {
        return cv::boundingRect(arg1);
    }
    
    void boxFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, Size arg4, Point arg5, bool arg6, int arg7) {
        return cv::boxFilter(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void boxPoints_wrapper(RotatedRect arg1, cv::Mat& arg2) {
        return cv::boxPoints(arg1, arg2);
    }
    
    void calcBackProject_wrapper(const std::vector<cv::Mat>& arg1, const std::vector<int>& arg2, const cv::Mat& arg3, cv::Mat& arg4, const std::vector<float>& arg5, double arg6) {
        return cv::calcBackProject(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void calcCovarMatrix_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::calcCovarMatrix(arg1, arg2, arg3, arg4, arg5);
    }
    
    void calcHist_wrapper(const std::vector<cv::Mat>& arg1, const std::vector<int>& arg2, const cv::Mat& arg3, cv::Mat& arg4, const std::vector<int>& arg5, const std::vector<float>& arg6, bool arg7) {
        return cv::calcHist(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void cartToPolar_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, bool arg5) {
        return cv::cartToPolar(arg1, arg2, arg3, arg4, arg5);
    }
    
    void circle_wrapper(cv::Mat& arg1, Point arg2, int arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::circle(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    bool clipLine_wrapper(Rect arg1, Point& arg2, Point& arg3) {
        return cv::clipLine(arg1, arg2, arg3);
    }
    
    void compare_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::compare(arg1, arg2, arg3, arg4);
    }
    
    double compareHist_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, int arg3) {
        return cv::compareHist(arg1, arg2, arg3);
    }
    
    void completeSymm_wrapper(cv::Mat& arg1, bool arg2) {
        return cv::completeSymm(arg1, arg2);
    }
    
    int connectedComponents_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::connectedComponents(arg1, arg2, arg3, arg4);
    }
    
    int connectedComponentsWithStats_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, int arg6) {
        return cv::connectedComponentsWithStats(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    double contourArea_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::contourArea(arg1, arg2);
    }
    
    void convertMaps_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, bool arg6) {
        return cv::convertMaps(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void convertScaleAbs_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4) {
        return cv::convertScaleAbs(arg1, arg2, arg3, arg4);
    }
    
    void convexHull_wrapper(const cv::Mat& arg1, cv::Mat& arg2, bool arg3, bool arg4) {
        return cv::convexHull(arg1, arg2, arg3, arg4);
    }
    
    void convexityDefects_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::convexityDefects(arg1, arg2, arg3);
    }
    
    void copyMakeBorder_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, int arg6, int arg7, const Scalar& arg8) {
        return cv::copyMakeBorder(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    void cornerEigenValsAndVecs_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::cornerEigenValsAndVecs(arg1, arg2, arg3, arg4, arg5);
    }
    
    void cornerHarris_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, double arg5, int arg6) {
        return cv::cornerHarris(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void cornerMinEigenVal_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::cornerMinEigenVal(arg1, arg2, arg3, arg4, arg5);
    }
    
    void cornerSubPix_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, Size arg4, TermCriteria arg5) {
        return cv::cornerSubPix(arg1, arg2, arg3, arg4, arg5);
    }
    
    int countNonZero_wrapper(const cv::Mat& arg1) {
        return cv::countNonZero(arg1);
    }
    
    void createHanningWindow_wrapper(cv::Mat& arg1, Size arg2, int arg3) {
        return cv::createHanningWindow(arg1, arg2, arg3);
    }
    
    void cvtColor_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::cvtColor(arg1, arg2, arg3, arg4);
    }
    
    void dct_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::dct(arg1, arg2, arg3);
    }
    
    void demosaicing_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::demosaicing(arg1, arg2, arg3, arg4);
    }
    
    double determinant_wrapper(const cv::Mat& arg1) {
        return cv::determinant(arg1);
    }
    
    void dft_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::dft(arg1, arg2, arg3, arg4);
    }
    
    void dilate_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Point arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::dilate(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void distanceTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::distanceTransform(arg1, arg2, arg3, arg4, arg5);
    }
    
    void distanceTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5, int arg6) {
        return cv::distanceTransform(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void divide_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5) {
        return cv::divide(arg1, arg2, arg3, arg4, arg5);
    }
    
    void divide_wrapper1(double arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::divide(arg1, arg2, arg3, arg4);
    }
    
    void drawContours_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, int arg3, const Scalar& arg4, int arg5, int arg6, const cv::Mat& arg7, int arg8, Point arg9) {
        return cv::drawContours(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void drawKeypoints_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, cv::Mat& arg3, const Scalar& arg4, int arg5) {
        return cv::drawKeypoints(arg1, arg2, arg3, arg4, arg5);
    }
    
    void drawMarker_wrapper(Mat& arg1, Point arg2, const Scalar& arg3, int arg4, int arg5, int arg6, int arg7) {
        return cv::drawMarker(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void drawMatches_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, const cv::Mat& arg3, const std::vector<KeyPoint>& arg4, const std::vector<DMatch>& arg5, cv::Mat& arg6, const Scalar& arg7, const Scalar& arg8, const std::vector<char>& arg9, int arg10) {
        return cv::drawMatches(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }
    
    void drawMatches_wrapper(const cv::Mat& arg1, const std::vector<KeyPoint>& arg2, const cv::Mat& arg3, const std::vector<KeyPoint>& arg4, const std::vector<std::vector<DMatch> >& arg5, cv::Mat& arg6, const Scalar& arg7, const Scalar& arg8, const std::vector<std::vector<char> >& arg9, int arg10) {
        return cv::drawMatches(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }
    
    bool eigen_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3) {
        return cv::eigen(arg1, arg2, arg3);
    }
    
    void ellipse_wrapper(cv::Mat& arg1, Point arg2, Size arg3, double arg4, double arg5, double arg6, const Scalar& arg7, int arg8, int arg9, int arg10) {
        return cv::ellipse(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }
    
    void ellipse_wrapper1(cv::Mat& arg1, const RotatedRect& arg2, const Scalar& arg3, int arg4, int arg5) {
        return cv::ellipse(arg1, arg2, arg3, arg4, arg5);
    }
    
    void ellipse2Poly_wrapper(Point arg1, Size arg2, int arg3, int arg4, int arg5, int arg6, std::vector<Point>& arg7) {
        return cv::ellipse2Poly(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void equalizeHist_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::equalizeHist(arg1, arg2);
    }
    
    void erode_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Point arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::erode(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void exp_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::exp(arg1, arg2);
    }
    
    void extractChannel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::extractChannel(arg1, arg2, arg3);
    }
    
    void fillConvexPoly_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const Scalar& arg3, int arg4, int arg5) {
        return cv::fillConvexPoly(arg1, arg2, arg3, arg4, arg5);
    }
    
    void fillPoly_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, const Scalar& arg3, int arg4, int arg5, Point arg6) {
        return cv::fillPoly(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void filter2D_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, Point arg5, double arg6, int arg7) {
        return cv::filter2D(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void findContours_wrapper(cv::Mat& arg1, std::vector<cv::Mat>& arg2, cv::Mat& arg3, int arg4, int arg5, Point arg6) {
        return cv::findContours(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void findNonZero_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::findNonZero(arg1, arg2);
    }
    
    RotatedRect fitEllipse_wrapper(const cv::Mat& arg1) {
        return cv::fitEllipse(arg1);
    }
    
    void fitLine_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, double arg6) {
        return cv::fitLine(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void flip_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::flip(arg1, arg2, arg3);
    }
    
    void gemm_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, double arg3, const cv::Mat& arg4, double arg5, cv::Mat& arg6, int arg7) {
        return cv::gemm(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    Mat getAffineTransform_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::getAffineTransform(arg1, arg2);
    }
    
    Mat getDefaultNewCameraMatrix_wrapper(const cv::Mat& arg1, Size arg2, bool arg3) {
        return cv::getDefaultNewCameraMatrix(arg1, arg2, arg3);
    }
    
    void getDerivKernels_wrapper(cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5, bool arg6, int arg7) {
        return cv::getDerivKernels(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    Mat getGaborKernel_wrapper(Size arg1, double arg2, double arg3, double arg4, double arg5, double arg6, int arg7) {
        return cv::getGaborKernel(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    Mat getGaussianKernel_wrapper(int arg1, double arg2, int arg3) {
        return cv::getGaussianKernel(arg1, arg2, arg3);
    }
    
    int getOptimalDFTSize_wrapper(int arg1) {
        return cv::getOptimalDFTSize(arg1);
    }
    
    Mat getPerspectiveTransform_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::getPerspectiveTransform(arg1, arg2);
    }
    
    void getRectSubPix_wrapper(const cv::Mat& arg1, Size arg2, Point2f arg3, cv::Mat& arg4, int arg5) {
        return cv::getRectSubPix(arg1, arg2, arg3, arg4, arg5);
    }
    
    Mat getRotationMatrix2D_wrapper(Point2f arg1, double arg2, double arg3) {
        return cv::getRotationMatrix2D(arg1, arg2, arg3);
    }
    
    Mat getStructuringElement_wrapper(int arg1, Size arg2, Point arg3) {
        return cv::getStructuringElement(arg1, arg2, arg3);
    }
    
    Size getTextSize_wrapper(const std::string& arg1, int arg2, double arg3, int arg4, int* arg5) {
        return cv::getTextSize(arg1, arg2, arg3, arg4, arg5);
    }
    
    void goodFeaturesToTrack_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, double arg4, double arg5, const cv::Mat& arg6, int arg7, bool arg8, double arg9) {
        return cv::goodFeaturesToTrack(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void grabCut_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Rect arg3, cv::Mat& arg4, cv::Mat& arg5, int arg6, int arg7) {
        return cv::grabCut(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void hconcat_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::hconcat(arg1, arg2);
    }
    
    void idct_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::idct(arg1, arg2, arg3);
    }
    
    void idft_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::idft(arg1, arg2, arg3, arg4);
    }
    
    void inRange_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::inRange(arg1, arg2, arg3, arg4);
    }
    
    void initUndistortRectifyMap_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, Size arg5, int arg6, cv::Mat& arg7, cv::Mat& arg8) {
        return cv::initUndistortRectifyMap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    float initWideAngleProjMap_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, Size arg3, int arg4, int arg5, cv::Mat& arg6, cv::Mat& arg7, int arg8, double arg9) {
        return cv::initWideAngleProjMap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void insertChannel_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::insertChannel(arg1, arg2, arg3);
    }
    
    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::integral(arg1, arg2, arg3);
    }
    
    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5);
    }
    
    void integral_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, int arg5, int arg6) {
        return cv::integral(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    float intersectConvexConvex_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, bool arg4) {
        return cv::intersectConvexConvex(arg1, arg2, arg3, arg4);
    }
    
    double invert_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::invert(arg1, arg2, arg3);
    }
    
    void invertAffineTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::invertAffineTransform(arg1, arg2);
    }
    
    bool isContourConvex_wrapper(const cv::Mat& arg1) {
        return cv::isContourConvex(arg1);
    }
    
    double kmeans_wrapper(const cv::Mat& arg1, int arg2, cv::Mat& arg3, TermCriteria arg4, int arg5, int arg6, cv::Mat& arg7) {
        return cv::kmeans(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void line_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::line(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void linearPolar_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Point2f arg3, double arg4, int arg5) {
        return cv::linearPolar(arg1, arg2, arg3, arg4, arg5);
    }
    
    void log_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::log(arg1, arg2);
    }
    
    void logPolar_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Point2f arg3, double arg4, int arg5) {
        return cv::logPolar(arg1, arg2, arg3, arg4, arg5);
    }
    
    void magnitude_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::magnitude(arg1, arg2, arg3);
    }
    
    double matchShapes_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, int arg3, double arg4) {
        return cv::matchShapes(arg1, arg2, arg3, arg4);
    }
    
    void matchTemplate_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, const cv::Mat& arg5) {
        return cv::matchTemplate(arg1, arg2, arg3, arg4, arg5);
    }
    
    void max_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::max(arg1, arg2, arg3);
    }
    
    Scalar mean_wrapper(const cv::Mat& arg1, const cv::Mat& arg2) {
        return cv::mean(arg1, arg2);
    }
    
    void meanStdDev_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4) {
        return cv::meanStdDev(arg1, arg2, arg3, arg4);
    }
    
    void medianBlur_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::medianBlur(arg1, arg2, arg3);
    }
    
    void merge_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::merge(arg1, arg2);
    }
    
    void min_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3) {
        return cv::min(arg1, arg2, arg3);
    }
    
    RotatedRect minAreaRect_wrapper(const cv::Mat& arg1) {
        return cv::minAreaRect(arg1);
    }
    
    double minEnclosingTriangle_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::minEnclosingTriangle(arg1, arg2);
    }
    
    void mixChannels_wrapper(const std::vector<cv::Mat>& arg1, InputOutputArrayOfArrays arg2, const std::vector<int>& arg3) {
        return cv::mixChannels(arg1, arg2, arg3);
    }
    
    Moments moments_wrapper(const cv::Mat& arg1, bool arg2) {
        return cv::moments(arg1, arg2);
    }
    
    void morphologyEx_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, Point arg5, int arg6, int arg7, const Scalar& arg8) {
        return cv::morphologyEx(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    void mulSpectrums_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4, bool arg5) {
        return cv::mulSpectrums(arg1, arg2, arg3, arg4, arg5);
    }
    
    void mulTransposed_wrapper(const cv::Mat& arg1, cv::Mat& arg2, bool arg3, const cv::Mat& arg4, double arg5, int arg6) {
        return cv::mulTransposed(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void multiply_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, double arg4, int arg5) {
        return cv::multiply(arg1, arg2, arg3, arg4, arg5);
    }
    
    double norm_wrapper(const cv::Mat& arg1, int arg2, const cv::Mat& arg3) {
        return cv::norm(arg1, arg2, arg3);
    }
    
    double norm_wrapper1(const cv::Mat& arg1, const cv::Mat& arg2, int arg3, const cv::Mat& arg4) {
        return cv::norm(arg1, arg2, arg3, arg4);
    }
    
    void normalize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, int arg6, const cv::Mat& arg7) {
        return cv::normalize(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void patchNaNs_wrapper(cv::Mat& arg1, double arg2) {
        return cv::patchNaNs(arg1, arg2);
    }
    
    void perspectiveTransform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::perspectiveTransform(arg1, arg2, arg3);
    }
    
    void phase_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, bool arg4) {
        return cv::phase(arg1, arg2, arg3, arg4);
    }
    
    double pointPolygonTest_wrapper(const cv::Mat& arg1, Point2f arg2, bool arg3) {
        return cv::pointPolygonTest(arg1, arg2, arg3);
    }
    
    void polarToCart_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, bool arg5) {
        return cv::polarToCart(arg1, arg2, arg3, arg4, arg5);
    }
    
    void polylines_wrapper(cv::Mat& arg1, const std::vector<cv::Mat>& arg2, bool arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::polylines(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void pow_wrapper(const cv::Mat& arg1, double arg2, cv::Mat& arg3) {
        return cv::pow(arg1, arg2, arg3);
    }
    
    void preCornerDetect_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4) {
        return cv::preCornerDetect(arg1, arg2, arg3, arg4);
    }
    
    void putText_wrapper(cv::Mat& arg1, const std::string& arg2, Point arg3, int arg4, double arg5, Scalar arg6, int arg7, int arg8, bool arg9) {
        return cv::putText(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void pyrDown_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const Size& arg3, int arg4) {
        return cv::pyrDown(arg1, arg2, arg3, arg4);
    }
    
    void pyrMeanShiftFiltering_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5, TermCriteria arg6) {
        return cv::pyrMeanShiftFiltering(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void pyrUp_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const Size& arg3, int arg4) {
        return cv::pyrUp(arg1, arg2, arg3, arg4);
    }
    
    void randn_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::randn(arg1, arg2, arg3);
    }
    
    void randu_wrapper(cv::Mat& arg1, const cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::randu(arg1, arg2, arg3);
    }
    
    void rectangle_wrapper(cv::Mat& arg1, Point arg2, Point arg3, const Scalar& arg4, int arg5, int arg6, int arg7) {
        return cv::rectangle(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void reduce_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, int arg4, int arg5) {
        return cv::reduce(arg1, arg2, arg3, arg4, arg5);
    }
    
    void remap_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::remap(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void repeat_wrapper(const cv::Mat& arg1, int arg2, int arg3, cv::Mat& arg4) {
        return cv::repeat(arg1, arg2, arg3, arg4);
    }
    
    void resize_wrapper(const cv::Mat& arg1, cv::Mat& arg2, Size arg3, double arg4, double arg5, int arg6) {
        return cv::resize(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    int rotatedRectangleIntersection_wrapper(const RotatedRect& arg1, const RotatedRect& arg2, cv::Mat& arg3) {
        return cv::rotatedRectangleIntersection(arg1, arg2, arg3);
    }
    
    void scaleAdd_wrapper(const cv::Mat& arg1, double arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return cv::scaleAdd(arg1, arg2, arg3, arg4);
    }
    
    void sepFilter2D_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, const cv::Mat& arg4, const cv::Mat& arg5, Point arg6, double arg7, int arg8) {
        return cv::sepFilter2D(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    void setIdentity_wrapper(cv::Mat& arg1, const Scalar& arg2) {
        return cv::setIdentity(arg1, arg2);
    }
    
    bool solve_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, int arg4) {
        return cv::solve(arg1, arg2, arg3, arg4);
    }
    
    int solveCubic_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::solveCubic(arg1, arg2);
    }
    
    double solvePoly_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::solvePoly(arg1, arg2, arg3);
    }
    
    void sort_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::sort(arg1, arg2, arg3);
    }
    
    void sortIdx_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3) {
        return cv::sortIdx(arg1, arg2, arg3);
    }
    
    void spatialGradient_wrapper(const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, int arg4, int arg5) {
        return cv::spatialGradient(arg1, arg2, arg3, arg4, arg5);
    }
    
    void split_wrapper(const cv::Mat& arg1, std::vector<cv::Mat>& arg2) {
        return cv::split(arg1, arg2);
    }
    
    void sqrBoxFilter_wrapper(const cv::Mat& arg1, cv::Mat& arg2, int arg3, Size arg4, Point arg5, bool arg6, int arg7) {
        return cv::sqrBoxFilter(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void sqrt_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::sqrt(arg1, arg2);
    }
    
    void subtract_wrapper(const cv::Mat& arg1, const cv::Mat& arg2, cv::Mat& arg3, const cv::Mat& arg4, int arg5) {
        return cv::subtract(arg1, arg2, arg3, arg4, arg5);
    }
    
    Scalar sum_wrapper(const cv::Mat& arg1) {
        return cv::sum(arg1);
    }
    
    double threshold_wrapper(const cv::Mat& arg1, cv::Mat& arg2, double arg3, double arg4, int arg5) {
        return cv::threshold(arg1, arg2, arg3, arg4, arg5);
    }
    
    Scalar trace_wrapper(const cv::Mat& arg1) {
        return cv::trace(arg1);
    }
    
    void transform_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3) {
        return cv::transform(arg1, arg2, arg3);
    }
    
    void transpose_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::transpose(arg1, arg2);
    }
    
    void undistort_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, const cv::Mat& arg5) {
        return cv::undistort(arg1, arg2, arg3, arg4, arg5);
    }
    
    void undistortPoints_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, const cv::Mat& arg4, const cv::Mat& arg5, const cv::Mat& arg6) {
        return cv::undistortPoints(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void vconcat_wrapper(const std::vector<cv::Mat>& arg1, cv::Mat& arg2) {
        return cv::vconcat(arg1, arg2);
    }
    
    void warpAffine_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Size arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::warpAffine(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void warpPerspective_wrapper(const cv::Mat& arg1, cv::Mat& arg2, const cv::Mat& arg3, Size arg4, int arg5, int arg6, const Scalar& arg7) {
        return cv::warpPerspective(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void watershed_wrapper(const cv::Mat& arg1, cv::Mat& arg2) {
        return cv::watershed(arg1, arg2);
    }
    
    void finish_wrapper() {
        return cv::ocl::finish();
    }
    
    bool haveAmdBlas_wrapper() {
        return cv::ocl::haveAmdBlas();
    }
    
    bool haveAmdFft_wrapper() {
        return cv::ocl::haveAmdFft();
    }
    
    bool haveOpenCL_wrapper() {
        return cv::ocl::haveOpenCL();
    }
    
    void setUseOpenCL_wrapper(bool arg1) {
        return cv::ocl::setUseOpenCL(arg1);
    }
    
    bool useOpenCL_wrapper() {
        return cv::ocl::useOpenCL();
    }
    
    void BOWImgDescriptorExtractor_setVocabulary_wrapper(cv::BOWImgDescriptorExtractor& arg0 , const Mat& arg1) {
        return arg0.setVocabulary(arg1);
    }
    
    void BOWImgDescriptorExtractor_compute2_wrapper(cv::BOWImgDescriptorExtractor& arg0 , const Mat& arg1, std::vector<KeyPoint>& arg2, Mat& arg3) {
        return arg0.compute2(arg1, arg2, arg3);
    }
    
    void KeyPoint_convert_wrapper(cv::KeyPoint& arg0 , const std::vector<KeyPoint>& arg1, std::vector<Point2f>& arg2, const std::vector<int>& arg3) {
        return arg0.convert(arg1, arg2, arg3);
    }
    
    void KeyPoint_convert_wrapper1(cv::KeyPoint& arg0 , const std::vector<Point2f>& arg1, std::vector<KeyPoint>& arg2, float arg3, float arg4, int arg5, int arg6) {
        return arg0.convert(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    float KeyPoint_overlap_wrapper(cv::KeyPoint& arg0 , const KeyPoint& arg1, const KeyPoint& arg2) {
        return arg0.overlap(arg1, arg2);
    }
    
    int LineSegmentDetector_compareSegments_wrapper(cv::LineSegmentDetector& arg0 , const Size& arg1, const cv::Mat& arg2, const cv::Mat& arg3, cv::Mat& arg4) {
        return arg0.compareSegments(arg1, arg2, arg3, arg4);
    }
    
    void LineSegmentDetector_detect_wrapper(cv::LineSegmentDetector& arg0 , const cv::Mat& arg1, cv::Mat& arg2, cv::Mat& arg3, cv::Mat& arg4, cv::Mat& arg5) {
        return arg0.detect(arg1, arg2, arg3, arg4, arg5);
    }
    
    void LineSegmentDetector_drawSegments_wrapper(cv::LineSegmentDetector& arg0 , cv::Mat& arg1, const cv::Mat& arg2) {
        return arg0.drawSegments(arg1, arg2);
    }
    
    Ptr<LineSegmentDetector> _createLineSegmentDetector_wrapper(int arg1, double arg2, double arg3, double arg4, double arg5, double arg6, double arg7, int arg8) {
        return cv::createLineSegmentDetector(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
    
    Mat BOWKMeansTrainer_cluster_wrapper(cv::BOWKMeansTrainer& arg0 ) {
        return arg0.cluster();
    }
    
    Mat BOWKMeansTrainer_cluster_wrapper1(cv::BOWKMeansTrainer& arg0 , const Mat& arg1) {
        return arg0.cluster(arg1);
    }
    
    void CLAHE_setTilesGridSize_wrapper(cv::CLAHE& arg0 , Size arg1) {
        return arg0.setTilesGridSize(arg1);
    }
    
    Ptr<CLAHE> _createCLAHE_wrapper(double arg1, Size arg2) {
        return cv::createCLAHE(arg1, arg2);
    }
    
    void CLAHE_setClipLimit_wrapper(cv::CLAHE& arg0 , double arg1) {
        return arg0.setClipLimit(arg1);
    }
    
    void CLAHE_apply_wrapper(cv::CLAHE& arg0 , const cv::Mat& arg1, cv::Mat& arg2) {
        return arg0.apply(arg1, arg2);
    }
    
    int Subdiv2D_insert_wrapper(cv::Subdiv2D& arg0 , Point2f arg1) {
        return arg0.insert(arg1);
    }
    
    void Subdiv2D_insert_wrapper1(cv::Subdiv2D& arg0 , const std::vector<Point2f>& arg1) {
        return arg0.insert(arg1);
    }
    
    int Subdiv2D_edgeOrg_wrapper(cv::Subdiv2D& arg0 , int arg1, Point2f* arg2) {
        return arg0.edgeOrg(arg1, arg2);
    }
    
    int Subdiv2D_rotateEdge_wrapper(cv::Subdiv2D& arg0 , int arg1, int arg2) {
        return arg0.rotateEdge(arg1, arg2);
    }
    
    void Subdiv2D_initDelaunay_wrapper(cv::Subdiv2D& arg0 , Rect arg1) {
        return arg0.initDelaunay(arg1);
    }
    
    int Subdiv2D_getEdge_wrapper(cv::Subdiv2D& arg0 , int arg1, int arg2) {
        return arg0.getEdge(arg1, arg2);
    }
    
    void Subdiv2D_getTriangleList_wrapper(cv::Subdiv2D& arg0 , std::vector<Vec6f>& arg1) {
        return arg0.getTriangleList(arg1);
    }
    
    int Subdiv2D_nextEdge_wrapper(cv::Subdiv2D& arg0 , int arg1) {
        return arg0.nextEdge(arg1);
    }
    
    int Subdiv2D_edgeDst_wrapper(cv::Subdiv2D& arg0 , int arg1, Point2f* arg2) {
        return arg0.edgeDst(arg1, arg2);
    }
    
    void Subdiv2D_getEdgeList_wrapper(cv::Subdiv2D& arg0 , std::vector<Vec4f>& arg1) {
        return arg0.getEdgeList(arg1);
    }
    
    Point2f Subdiv2D_getVertex_wrapper(cv::Subdiv2D& arg0 , int arg1, int* arg2) {
        return arg0.getVertex(arg1, arg2);
    }
    
    void Subdiv2D_getVoronoiFacetList_wrapper(cv::Subdiv2D& arg0 , const std::vector<int>& arg1, std::vector<std::vector<Point2f> >& arg2, std::vector<Point2f>& arg3) {
        return arg0.getVoronoiFacetList(arg1, arg2, arg3);
    }
    
    int Subdiv2D_symEdge_wrapper(cv::Subdiv2D& arg0 , int arg1) {
        return arg0.symEdge(arg1);
    }
    
    int Subdiv2D_findNearest_wrapper(cv::Subdiv2D& arg0 , Point2f arg1, Point2f* arg2) {
        return arg0.findNearest(arg1, arg2);
    }
    
    Ptr<BRISK> BRISK_create_wrapper(int arg1, int arg2, float arg3) {
        return cv::BRISK::create(arg1, arg2, arg3);
    }
    
    Ptr<BRISK> BRISK_create_wrapper1(const std::vector<float> & arg1, const std::vector<int> & arg2, float arg3, float arg4, const std::vector<int>& arg5) {
        return cv::BRISK::create(arg1, arg2, arg3, arg4, arg5);
    }
    
    void KAZE_setExtended_wrapper(cv::KAZE& arg0 , bool arg1) {
        return arg0.setExtended(arg1);
    }
    
    void KAZE_setNOctaveLayers_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setNOctaveLayers(arg1);
    }
    
    void KAZE_setNOctaves_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setNOctaves(arg1);
    }
    
    Ptr<KAZE> KAZE_create_wrapper(bool arg1, bool arg2, float arg3, int arg4, int arg5, int arg6) {
        return cv::KAZE::create(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void KAZE_setUpright_wrapper(cv::KAZE& arg0 , bool arg1) {
        return arg0.setUpright(arg1);
    }
    
    void KAZE_setDiffusivity_wrapper(cv::KAZE& arg0 , int arg1) {
        return arg0.setDiffusivity(arg1);
    }
    
    void KAZE_setThreshold_wrapper(cv::KAZE& arg0 , double arg1) {
        return arg0.setThreshold(arg1);
    }
    
    std::string Algorithm_getDefaultName_wrapper(cv::Algorithm& arg0 ) {
        return arg0.getDefaultName();
    }
    
    void Algorithm_save_wrapper(cv::Algorithm& arg0 , const std::string& arg1) {
        return arg0.save(arg1);
    }
    
    void Feature2D_detect_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, std::vector<KeyPoint>& arg2, const cv::Mat& arg3) {
        return arg0.detect(arg1, arg2, arg3);
    }
    
    void Feature2D_compute_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, std::vector<KeyPoint>& arg2, cv::Mat& arg3) {
        return arg0.compute(arg1, arg2, arg3);
    }
    
    void Feature2D_detectAndCompute_wrapper(cv::Feature2D& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<KeyPoint>& arg3, cv::Mat& arg4, bool arg5) {
        return arg0.detectAndCompute(arg1, arg2, arg3, arg4, arg5);
    }
    
    void GFTTDetector_setHarrisDetector_wrapper(cv::GFTTDetector& arg0 , bool arg1) {
        return arg0.setHarrisDetector(arg1);
    }
    
    void GFTTDetector_setBlockSize_wrapper(cv::GFTTDetector& arg0 , int arg1) {
        return arg0.setBlockSize(arg1);
    }
    
    Ptr<GFTTDetector> GFTTDetector_create_wrapper(int arg1, double arg2, double arg3, int arg4, bool arg5, double arg6) {
        return cv::GFTTDetector::create(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void GFTTDetector_setQualityLevel_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setQualityLevel(arg1);
    }
    
    void GFTTDetector_setMaxFeatures_wrapper(cv::GFTTDetector& arg0 , int arg1) {
        return arg0.setMaxFeatures(arg1);
    }
    
    void GFTTDetector_setK_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setK(arg1);
    }
    
    void GFTTDetector_setMinDistance_wrapper(cv::GFTTDetector& arg0 , double arg1) {
        return arg0.setMinDistance(arg1);
    }
    
    Ptr<DescriptorMatcher> DescriptorMatcher_create_wrapper(const std::string& arg1) {
        return cv::DescriptorMatcher::create(arg1);
    }
    
    void DescriptorMatcher_knnMatch_wrapper(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<std::vector<DMatch> >& arg3, int arg4, const cv::Mat& arg5, bool arg6) {
        return arg0.knnMatch(arg1, arg2, arg3, arg4, arg5, arg6);
    }
    
    void DescriptorMatcher_knnMatch_wrapper1(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, std::vector<std::vector<DMatch> >& arg2, int arg3, const std::vector<cv::Mat>& arg4, bool arg5) {
        return arg0.knnMatch(arg1, arg2, arg3, arg4, arg5);
    }
    
    void DescriptorMatcher_add_wrapper(cv::DescriptorMatcher& arg0 , const std::vector<cv::Mat>& arg1) {
        return arg0.add(arg1);
    }
    
    void DescriptorMatcher_match_wrapper(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, const cv::Mat& arg2, std::vector<DMatch>& arg3, const cv::Mat& arg4) {
        return arg0.match(arg1, arg2, arg3, arg4);
    }
    
    void DescriptorMatcher_match_wrapper1(cv::DescriptorMatcher& arg0 , const cv::Mat& arg1, std::vector<DMatch>& arg2, const std::vector<cv::Mat>& arg3) {
        return arg0.match(arg1, arg2, arg3);
    }
    
    void MSER_setMinArea_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setMinArea(arg1);
    }
    
    Ptr<MSER> MSER_create_wrapper(int arg1, int arg2, int arg3, double arg4, double arg5, int arg6, double arg7, double arg8, int arg9) {
        return cv::MSER::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void MSER_setMaxArea_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setMaxArea(arg1);
    }
    
    void MSER_setPass2Only_wrapper(cv::MSER& arg0 , bool arg1) {
        return arg0.setPass2Only(arg1);
    }
    
    void MSER_detectRegions_wrapper(cv::MSER& arg0 , const cv::Mat& arg1, std::vector<std::vector<Point> >& arg2, std::vector<Rect>& arg3) {
        return arg0.detectRegions(arg1, arg2, arg3);
    }
    
    void MSER_setDelta_wrapper(cv::MSER& arg0 , int arg1) {
        return arg0.setDelta(arg1);
    }
    
    Ptr<SimpleBlobDetector> SimpleBlobDetector_create_wrapper(const SimpleBlobDetector::Params & arg1) {
        return cv::SimpleBlobDetector::create(arg1);
    }
    
    Ptr<AgastFeatureDetector> AgastFeatureDetector_create_wrapper(int arg1, bool arg2, int arg3) {
        return cv::AgastFeatureDetector::create(arg1, arg2, arg3);
    }
    
    void AgastFeatureDetector_setNonmaxSuppression_wrapper(cv::AgastFeatureDetector& arg0 , bool arg1) {
        return arg0.setNonmaxSuppression(arg1);
    }
    
    void AgastFeatureDetector_setThreshold_wrapper(cv::AgastFeatureDetector& arg0 , int arg1) {
        return arg0.setThreshold(arg1);
    }
    
    void AgastFeatureDetector_setType_wrapper(cv::AgastFeatureDetector& arg0 , int arg1) {
        return arg0.setType(arg1);
    }
    
    Ptr<FastFeatureDetector> FastFeatureDetector_create_wrapper(int arg1, bool arg2, int arg3) {
        return cv::FastFeatureDetector::create(arg1, arg2, arg3);
    }
    
    void FastFeatureDetector_setNonmaxSuppression_wrapper(cv::FastFeatureDetector& arg0 , bool arg1) {
        return arg0.setNonmaxSuppression(arg1);
    }
    
    void FastFeatureDetector_setThreshold_wrapper(cv::FastFeatureDetector& arg0 , int arg1) {
        return arg0.setThreshold(arg1);
    }
    
    void FastFeatureDetector_setType_wrapper(cv::FastFeatureDetector& arg0 , int arg1) {
        return arg0.setType(arg1);
    }
    
    void AKAZE_setNOctaveLayers_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setNOctaveLayers(arg1);
    }
    
    void AKAZE_setDescriptorType_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorType(arg1);
    }
    
    void AKAZE_setNOctaves_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setNOctaves(arg1);
    }
    
    Ptr<AKAZE> AKAZE_create_wrapper(int arg1, int arg2, int arg3, float arg4, int arg5, int arg6, int arg7) {
        return cv::AKAZE::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    
    void AKAZE_setDescriptorChannels_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorChannels(arg1);
    }
    
    void AKAZE_setThreshold_wrapper(cv::AKAZE& arg0 , double arg1) {
        return arg0.setThreshold(arg1);
    }
    
    void AKAZE_setDescriptorSize_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDescriptorSize(arg1);
    }
    
    void AKAZE_setDiffusivity_wrapper(cv::AKAZE& arg0 , int arg1) {
        return arg0.setDiffusivity(arg1);
    }
    
    Mat BOWTrainer_cluster_wrapper(cv::BOWTrainer& arg0 ) {
        return arg0.cluster();
    }
    
    Mat BOWTrainer_cluster_wrapper1(cv::BOWTrainer& arg0 , const Mat& arg1) {
        return arg0.cluster(arg1);
    }
    
    void BOWTrainer_add_wrapper(cv::BOWTrainer& arg0 , const Mat& arg1) {
        return arg0.add(arg1);
    }
    
    void ORB_setEdgeThreshold_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setEdgeThreshold(arg1);
    }
    
    void ORB_setFirstLevel_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setFirstLevel(arg1);
    }
    
    Ptr<ORB> ORB_create_wrapper(int arg1, float arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, int arg9) {
        return cv::ORB::create(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
    
    void ORB_setMaxFeatures_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setMaxFeatures(arg1);
    }
    
    void ORB_setNLevels_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setNLevels(arg1);
    }
    
    void ORB_setFastThreshold_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setFastThreshold(arg1);
    }
    
    void ORB_setPatchSize_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setPatchSize(arg1);
    }
    
    void ORB_setWTA_K_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setWTA_K(arg1);
    }
    
    void ORB_setScaleFactor_wrapper(cv::ORB& arg0 , double arg1) {
        return arg0.setScaleFactor(arg1);
    }
    
    void ORB_setScoreType_wrapper(cv::ORB& arg0 , int arg1) {
        return arg0.setScoreType(arg1);
    }
    
}

EMSCRIPTEN_BINDINGS(testBinding) {

    function("cvtColor", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::cvtColor_wrapper));

    emscripten::class_<cv::Algorithm >("Algorithm")
        .function("getDefaultName", select_overload<std::string(cv::Algorithm&)>(&Wrappers::Algorithm_getDefaultName_wrapper))
        .function("clear", select_overload<void()>(&cv::Algorithm::clear))
        .function("save", select_overload<void(cv::Algorithm&,const std::string&)>(&Wrappers::Algorithm_save_wrapper));

    emscripten::enum_<ColorConversionCodes>("ColorConversionCodes")
        .value("COLOR_BGR2BGRA", ColorConversionCodes::COLOR_BGR2BGRA)
        .value("COLOR_RGB2RGBA", ColorConversionCodes::COLOR_RGB2RGBA)
        .value("COLOR_BGRA2BGR", ColorConversionCodes::COLOR_BGRA2BGR)
        .value("COLOR_RGBA2RGB", ColorConversionCodes::COLOR_RGBA2RGB)
        .value("COLOR_BGR2RGBA", ColorConversionCodes::COLOR_BGR2RGBA)
        .value("COLOR_RGB2BGRA", ColorConversionCodes::COLOR_RGB2BGRA)
        .value("COLOR_RGBA2BGR", ColorConversionCodes::COLOR_RGBA2BGR)
        .value("COLOR_BGRA2RGB", ColorConversionCodes::COLOR_BGRA2RGB)
        .value("COLOR_BGR2RGB", ColorConversionCodes::COLOR_BGR2RGB)
        .value("COLOR_RGB2BGR", ColorConversionCodes::COLOR_RGB2BGR)
        .value("COLOR_BGRA2RGBA", ColorConversionCodes::COLOR_BGRA2RGBA)
        .value("COLOR_RGBA2BGRA", ColorConversionCodes::COLOR_RGBA2BGRA)
        .value("COLOR_BGR2GRAY", ColorConversionCodes::COLOR_BGR2GRAY)
        .value("COLOR_RGB2GRAY", ColorConversionCodes::COLOR_RGB2GRAY)
        .value("COLOR_GRAY2BGR", ColorConversionCodes::COLOR_GRAY2BGR)
        .value("COLOR_GRAY2RGB", ColorConversionCodes::COLOR_GRAY2RGB)
        .value("COLOR_GRAY2BGRA", ColorConversionCodes::COLOR_GRAY2BGRA)
        .value("COLOR_GRAY2RGBA", ColorConversionCodes::COLOR_GRAY2RGBA)
        .value("COLOR_BGRA2GRAY", ColorConversionCodes::COLOR_BGRA2GRAY)
        .value("COLOR_RGBA2GRAY", ColorConversionCodes::COLOR_RGBA2GRAY)
        .value("COLOR_BGR2BGR565", ColorConversionCodes::COLOR_BGR2BGR565)
        .value("COLOR_RGB2BGR565", ColorConversionCodes::COLOR_RGB2BGR565)
        .value("COLOR_BGR5652BGR", ColorConversionCodes::COLOR_BGR5652BGR)
        .value("COLOR_BGR5652RGB", ColorConversionCodes::COLOR_BGR5652RGB)
        .value("COLOR_BGRA2BGR565", ColorConversionCodes::COLOR_BGRA2BGR565)
        .value("COLOR_RGBA2BGR565", ColorConversionCodes::COLOR_RGBA2BGR565)
        .value("COLOR_BGR5652BGRA", ColorConversionCodes::COLOR_BGR5652BGRA)
        .value("COLOR_BGR5652RGBA", ColorConversionCodes::COLOR_BGR5652RGBA)
        .value("COLOR_GRAY2BGR565", ColorConversionCodes::COLOR_GRAY2BGR565)
        .value("COLOR_BGR5652GRAY", ColorConversionCodes::COLOR_BGR5652GRAY)
        .value("COLOR_BGR2BGR555", ColorConversionCodes::COLOR_BGR2BGR555)
        .value("COLOR_RGB2BGR555", ColorConversionCodes::COLOR_RGB2BGR555)
        .value("COLOR_BGR5552BGR", ColorConversionCodes::COLOR_BGR5552BGR)
        .value("COLOR_BGR5552RGB", ColorConversionCodes::COLOR_BGR5552RGB)
        .value("COLOR_BGRA2BGR555", ColorConversionCodes::COLOR_BGRA2BGR555)
        .value("COLOR_RGBA2BGR555", ColorConversionCodes::COLOR_RGBA2BGR555)
        .value("COLOR_BGR5552BGRA", ColorConversionCodes::COLOR_BGR5552BGRA)
        .value("COLOR_BGR5552RGBA", ColorConversionCodes::COLOR_BGR5552RGBA)
        .value("COLOR_GRAY2BGR555", ColorConversionCodes::COLOR_GRAY2BGR555)
        .value("COLOR_BGR5552GRAY", ColorConversionCodes::COLOR_BGR5552GRAY)
        .value("COLOR_BGR2XYZ", ColorConversionCodes::COLOR_BGR2XYZ)
        .value("COLOR_RGB2XYZ", ColorConversionCodes::COLOR_RGB2XYZ)
        .value("COLOR_XYZ2BGR", ColorConversionCodes::COLOR_XYZ2BGR)
        .value("COLOR_XYZ2RGB", ColorConversionCodes::COLOR_XYZ2RGB)
        .value("COLOR_BGR2YCrCb", ColorConversionCodes::COLOR_BGR2YCrCb)
        .value("COLOR_RGB2YCrCb", ColorConversionCodes::COLOR_RGB2YCrCb)
        .value("COLOR_YCrCb2BGR", ColorConversionCodes::COLOR_YCrCb2BGR)
        .value("COLOR_YCrCb2RGB", ColorConversionCodes::COLOR_YCrCb2RGB)
        .value("COLOR_BGR2HSV", ColorConversionCodes::COLOR_BGR2HSV)
        .value("COLOR_RGB2HSV", ColorConversionCodes::COLOR_RGB2HSV)
        .value("COLOR_BGR2Lab", ColorConversionCodes::COLOR_BGR2Lab)
        .value("COLOR_RGB2Lab", ColorConversionCodes::COLOR_RGB2Lab)
        .value("COLOR_BGR2Luv", ColorConversionCodes::COLOR_BGR2Luv)
        .value("COLOR_RGB2Luv", ColorConversionCodes::COLOR_RGB2Luv)
        .value("COLOR_BGR2HLS", ColorConversionCodes::COLOR_BGR2HLS)
        .value("COLOR_RGB2HLS", ColorConversionCodes::COLOR_RGB2HLS)
        .value("COLOR_HSV2BGR", ColorConversionCodes::COLOR_HSV2BGR)
        .value("COLOR_HSV2RGB", ColorConversionCodes::COLOR_HSV2RGB)
        .value("COLOR_Lab2BGR", ColorConversionCodes::COLOR_Lab2BGR)
        .value("COLOR_Lab2RGB", ColorConversionCodes::COLOR_Lab2RGB)
        .value("COLOR_Luv2BGR", ColorConversionCodes::COLOR_Luv2BGR)
        .value("COLOR_Luv2RGB", ColorConversionCodes::COLOR_Luv2RGB)
        .value("COLOR_HLS2BGR", ColorConversionCodes::COLOR_HLS2BGR)
        .value("COLOR_HLS2RGB", ColorConversionCodes::COLOR_HLS2RGB)
        .value("COLOR_BGR2HSV_FULL", ColorConversionCodes::COLOR_BGR2HSV_FULL)
        .value("COLOR_RGB2HSV_FULL", ColorConversionCodes::COLOR_RGB2HSV_FULL)
        .value("COLOR_BGR2HLS_FULL", ColorConversionCodes::COLOR_BGR2HLS_FULL)
        .value("COLOR_RGB2HLS_FULL", ColorConversionCodes::COLOR_RGB2HLS_FULL)
        .value("COLOR_HSV2BGR_FULL", ColorConversionCodes::COLOR_HSV2BGR_FULL)
        .value("COLOR_HSV2RGB_FULL", ColorConversionCodes::COLOR_HSV2RGB_FULL)
        .value("COLOR_HLS2BGR_FULL", ColorConversionCodes::COLOR_HLS2BGR_FULL)
        .value("COLOR_HLS2RGB_FULL", ColorConversionCodes::COLOR_HLS2RGB_FULL)
        .value("COLOR_LBGR2Lab", ColorConversionCodes::COLOR_LBGR2Lab)
        .value("COLOR_LRGB2Lab", ColorConversionCodes::COLOR_LRGB2Lab)
        .value("COLOR_LBGR2Luv", ColorConversionCodes::COLOR_LBGR2Luv)
        .value("COLOR_LRGB2Luv", ColorConversionCodes::COLOR_LRGB2Luv)
        .value("COLOR_Lab2LBGR", ColorConversionCodes::COLOR_Lab2LBGR)
        .value("COLOR_Lab2LRGB", ColorConversionCodes::COLOR_Lab2LRGB)
        .value("COLOR_Luv2LBGR", ColorConversionCodes::COLOR_Luv2LBGR)
        .value("COLOR_Luv2LRGB", ColorConversionCodes::COLOR_Luv2LRGB)
        .value("COLOR_BGR2YUV", ColorConversionCodes::COLOR_BGR2YUV)
        .value("COLOR_RGB2YUV", ColorConversionCodes::COLOR_RGB2YUV)
        .value("COLOR_YUV2BGR", ColorConversionCodes::COLOR_YUV2BGR)
        .value("COLOR_YUV2RGB", ColorConversionCodes::COLOR_YUV2RGB)
        .value("COLOR_YUV2RGB_NV12", ColorConversionCodes::COLOR_YUV2RGB_NV12)
        .value("COLOR_YUV2BGR_NV12", ColorConversionCodes::COLOR_YUV2BGR_NV12)
        .value("COLOR_YUV2RGB_NV21", ColorConversionCodes::COLOR_YUV2RGB_NV21)
        .value("COLOR_YUV2BGR_NV21", ColorConversionCodes::COLOR_YUV2BGR_NV21)
        .value("COLOR_YUV420sp2RGB", ColorConversionCodes::COLOR_YUV420sp2RGB)
        .value("COLOR_YUV420sp2BGR", ColorConversionCodes::COLOR_YUV420sp2BGR)
        .value("COLOR_YUV2RGBA_NV12", ColorConversionCodes::COLOR_YUV2RGBA_NV12)
        .value("COLOR_YUV2BGRA_NV12", ColorConversionCodes::COLOR_YUV2BGRA_NV12)
        .value("COLOR_YUV2RGBA_NV21", ColorConversionCodes::COLOR_YUV2RGBA_NV21)
        .value("COLOR_YUV2BGRA_NV21", ColorConversionCodes::COLOR_YUV2BGRA_NV21)
        .value("COLOR_YUV420sp2RGBA", ColorConversionCodes::COLOR_YUV420sp2RGBA)
        .value("COLOR_YUV420sp2BGRA", ColorConversionCodes::COLOR_YUV420sp2BGRA)
        .value("COLOR_YUV2RGB_YV12", ColorConversionCodes::COLOR_YUV2RGB_YV12)
        .value("COLOR_YUV2BGR_YV12", ColorConversionCodes::COLOR_YUV2BGR_YV12)
        .value("COLOR_YUV2RGB_IYUV", ColorConversionCodes::COLOR_YUV2RGB_IYUV)
        .value("COLOR_YUV2BGR_IYUV", ColorConversionCodes::COLOR_YUV2BGR_IYUV)
        .value("COLOR_YUV2RGB_I420", ColorConversionCodes::COLOR_YUV2RGB_I420)
        .value("COLOR_YUV2BGR_I420", ColorConversionCodes::COLOR_YUV2BGR_I420)
        .value("COLOR_YUV420p2RGB", ColorConversionCodes::COLOR_YUV420p2RGB)
        .value("COLOR_YUV420p2BGR", ColorConversionCodes::COLOR_YUV420p2BGR)
        .value("COLOR_YUV2RGBA_YV12", ColorConversionCodes::COLOR_YUV2RGBA_YV12)
        .value("COLOR_YUV2BGRA_YV12", ColorConversionCodes::COLOR_YUV2BGRA_YV12)
        .value("COLOR_YUV2RGBA_IYUV", ColorConversionCodes::COLOR_YUV2RGBA_IYUV)
        .value("COLOR_YUV2BGRA_IYUV", ColorConversionCodes::COLOR_YUV2BGRA_IYUV)
        .value("COLOR_YUV2RGBA_I420", ColorConversionCodes::COLOR_YUV2RGBA_I420)
        .value("COLOR_YUV2BGRA_I420", ColorConversionCodes::COLOR_YUV2BGRA_I420)
        .value("COLOR_YUV420p2RGBA", ColorConversionCodes::COLOR_YUV420p2RGBA)
        .value("COLOR_YUV420p2BGRA", ColorConversionCodes::COLOR_YUV420p2BGRA)
        .value("COLOR_YUV2GRAY_420", ColorConversionCodes::COLOR_YUV2GRAY_420)
        .value("COLOR_YUV2GRAY_NV21", ColorConversionCodes::COLOR_YUV2GRAY_NV21)
        .value("COLOR_YUV2GRAY_NV12", ColorConversionCodes::COLOR_YUV2GRAY_NV12)
        .value("COLOR_YUV2GRAY_YV12", ColorConversionCodes::COLOR_YUV2GRAY_YV12)
        .value("COLOR_YUV2GRAY_IYUV", ColorConversionCodes::COLOR_YUV2GRAY_IYUV)
        .value("COLOR_YUV2GRAY_I420", ColorConversionCodes::COLOR_YUV2GRAY_I420)
        .value("COLOR_YUV420sp2GRAY", ColorConversionCodes::COLOR_YUV420sp2GRAY)
        .value("COLOR_YUV420p2GRAY", ColorConversionCodes::COLOR_YUV420p2GRAY)
        .value("COLOR_YUV2RGB_UYVY", ColorConversionCodes::COLOR_YUV2RGB_UYVY)
        .value("COLOR_YUV2BGR_UYVY", ColorConversionCodes::COLOR_YUV2BGR_UYVY)
        .value("COLOR_YUV2RGB_Y422", ColorConversionCodes::COLOR_YUV2RGB_Y422)
        .value("COLOR_YUV2BGR_Y422", ColorConversionCodes::COLOR_YUV2BGR_Y422)
        .value("COLOR_YUV2RGB_UYNV", ColorConversionCodes::COLOR_YUV2RGB_UYNV)
        .value("COLOR_YUV2BGR_UYNV", ColorConversionCodes::COLOR_YUV2BGR_UYNV)
        .value("COLOR_YUV2RGBA_UYVY", ColorConversionCodes::COLOR_YUV2RGBA_UYVY)
        .value("COLOR_YUV2BGRA_UYVY", ColorConversionCodes::COLOR_YUV2BGRA_UYVY)
        .value("COLOR_YUV2RGBA_Y422", ColorConversionCodes::COLOR_YUV2RGBA_Y422)
        .value("COLOR_YUV2BGRA_Y422", ColorConversionCodes::COLOR_YUV2BGRA_Y422)
        .value("COLOR_YUV2RGBA_UYNV", ColorConversionCodes::COLOR_YUV2RGBA_UYNV)
        .value("COLOR_YUV2BGRA_UYNV", ColorConversionCodes::COLOR_YUV2BGRA_UYNV)
        .value("COLOR_YUV2RGB_YUY2", ColorConversionCodes::COLOR_YUV2RGB_YUY2)
        .value("COLOR_YUV2BGR_YUY2", ColorConversionCodes::COLOR_YUV2BGR_YUY2)
        .value("COLOR_YUV2RGB_YVYU", ColorConversionCodes::COLOR_YUV2RGB_YVYU)
        .value("COLOR_YUV2BGR_YVYU", ColorConversionCodes::COLOR_YUV2BGR_YVYU)
        .value("COLOR_YUV2RGB_YUYV", ColorConversionCodes::COLOR_YUV2RGB_YUYV)
        .value("COLOR_YUV2BGR_YUYV", ColorConversionCodes::COLOR_YUV2BGR_YUYV)
        .value("COLOR_YUV2RGB_YUNV", ColorConversionCodes::COLOR_YUV2RGB_YUNV)
        .value("COLOR_YUV2BGR_YUNV", ColorConversionCodes::COLOR_YUV2BGR_YUNV)
        .value("COLOR_YUV2RGBA_YUY2", ColorConversionCodes::COLOR_YUV2RGBA_YUY2)
        .value("COLOR_YUV2BGRA_YUY2", ColorConversionCodes::COLOR_YUV2BGRA_YUY2)
        .value("COLOR_YUV2RGBA_YVYU", ColorConversionCodes::COLOR_YUV2RGBA_YVYU)
        .value("COLOR_YUV2BGRA_YVYU", ColorConversionCodes::COLOR_YUV2BGRA_YVYU)
        .value("COLOR_YUV2RGBA_YUYV", ColorConversionCodes::COLOR_YUV2RGBA_YUYV)
        .value("COLOR_YUV2BGRA_YUYV", ColorConversionCodes::COLOR_YUV2BGRA_YUYV)
        .value("COLOR_YUV2RGBA_YUNV", ColorConversionCodes::COLOR_YUV2RGBA_YUNV)
        .value("COLOR_YUV2BGRA_YUNV", ColorConversionCodes::COLOR_YUV2BGRA_YUNV)
        .value("COLOR_YUV2GRAY_UYVY", ColorConversionCodes::COLOR_YUV2GRAY_UYVY)
        .value("COLOR_YUV2GRAY_YUY2", ColorConversionCodes::COLOR_YUV2GRAY_YUY2)
        .value("COLOR_YUV2GRAY_Y422", ColorConversionCodes::COLOR_YUV2GRAY_Y422)
        .value("COLOR_YUV2GRAY_UYNV", ColorConversionCodes::COLOR_YUV2GRAY_UYNV)
        .value("COLOR_YUV2GRAY_YVYU", ColorConversionCodes::COLOR_YUV2GRAY_YVYU)
        .value("COLOR_YUV2GRAY_YUYV", ColorConversionCodes::COLOR_YUV2GRAY_YUYV)
        .value("COLOR_YUV2GRAY_YUNV", ColorConversionCodes::COLOR_YUV2GRAY_YUNV)
        .value("COLOR_RGBA2mRGBA", ColorConversionCodes::COLOR_RGBA2mRGBA)
        .value("COLOR_mRGBA2RGBA", ColorConversionCodes::COLOR_mRGBA2RGBA)
        .value("COLOR_RGB2YUV_I420", ColorConversionCodes::COLOR_RGB2YUV_I420)
        .value("COLOR_BGR2YUV_I420", ColorConversionCodes::COLOR_BGR2YUV_I420)
        .value("COLOR_RGB2YUV_IYUV", ColorConversionCodes::COLOR_RGB2YUV_IYUV)
        .value("COLOR_BGR2YUV_IYUV", ColorConversionCodes::COLOR_BGR2YUV_IYUV)
        .value("COLOR_RGBA2YUV_I420", ColorConversionCodes::COLOR_RGBA2YUV_I420)
        .value("COLOR_BGRA2YUV_I420", ColorConversionCodes::COLOR_BGRA2YUV_I420)
        .value("COLOR_RGBA2YUV_IYUV", ColorConversionCodes::COLOR_RGBA2YUV_IYUV)
        .value("COLOR_BGRA2YUV_IYUV", ColorConversionCodes::COLOR_BGRA2YUV_IYUV)
        .value("COLOR_RGB2YUV_YV12", ColorConversionCodes::COLOR_RGB2YUV_YV12)
        .value("COLOR_BGR2YUV_YV12", ColorConversionCodes::COLOR_BGR2YUV_YV12)
        .value("COLOR_RGBA2YUV_YV12", ColorConversionCodes::COLOR_RGBA2YUV_YV12)
        .value("COLOR_BGRA2YUV_YV12", ColorConversionCodes::COLOR_BGRA2YUV_YV12)
        .value("COLOR_BayerBG2BGR", ColorConversionCodes::COLOR_BayerBG2BGR)
        .value("COLOR_BayerGB2BGR", ColorConversionCodes::COLOR_BayerGB2BGR)
        .value("COLOR_BayerRG2BGR", ColorConversionCodes::COLOR_BayerRG2BGR)
        .value("COLOR_BayerGR2BGR", ColorConversionCodes::COLOR_BayerGR2BGR)
        .value("COLOR_BayerBG2RGB", ColorConversionCodes::COLOR_BayerBG2RGB)
        .value("COLOR_BayerGB2RGB", ColorConversionCodes::COLOR_BayerGB2RGB)
        .value("COLOR_BayerRG2RGB", ColorConversionCodes::COLOR_BayerRG2RGB)
        .value("COLOR_BayerGR2RGB", ColorConversionCodes::COLOR_BayerGR2RGB)
        .value("COLOR_BayerBG2GRAY", ColorConversionCodes::COLOR_BayerBG2GRAY)
        .value("COLOR_BayerGB2GRAY", ColorConversionCodes::COLOR_BayerGB2GRAY)
        .value("COLOR_BayerRG2GRAY", ColorConversionCodes::COLOR_BayerRG2GRAY)
        .value("COLOR_BayerGR2GRAY", ColorConversionCodes::COLOR_BayerGR2GRAY)
        .value("COLOR_BayerBG2BGR_VNG", ColorConversionCodes::COLOR_BayerBG2BGR_VNG)
        .value("COLOR_BayerGB2BGR_VNG", ColorConversionCodes::COLOR_BayerGB2BGR_VNG)
        .value("COLOR_BayerRG2BGR_VNG", ColorConversionCodes::COLOR_BayerRG2BGR_VNG)
        .value("COLOR_BayerGR2BGR_VNG", ColorConversionCodes::COLOR_BayerGR2BGR_VNG)
        .value("COLOR_BayerBG2RGB_VNG", ColorConversionCodes::COLOR_BayerBG2RGB_VNG)
        .value("COLOR_BayerGB2RGB_VNG", ColorConversionCodes::COLOR_BayerGB2RGB_VNG)
        .value("COLOR_BayerRG2RGB_VNG", ColorConversionCodes::COLOR_BayerRG2RGB_VNG)
        .value("COLOR_BayerGR2RGB_VNG", ColorConversionCodes::COLOR_BayerGR2RGB_VNG)
        .value("COLOR_BayerBG2BGR_EA", ColorConversionCodes::COLOR_BayerBG2BGR_EA)
        .value("COLOR_BayerGB2BGR_EA", ColorConversionCodes::COLOR_BayerGB2BGR_EA)
        .value("COLOR_BayerRG2BGR_EA", ColorConversionCodes::COLOR_BayerRG2BGR_EA)
        .value("COLOR_BayerGR2BGR_EA", ColorConversionCodes::COLOR_BayerGR2BGR_EA)
        .value("COLOR_BayerBG2RGB_EA", ColorConversionCodes::COLOR_BayerBG2RGB_EA)
        .value("COLOR_BayerGB2RGB_EA", ColorConversionCodes::COLOR_BayerGB2RGB_EA)
        .value("COLOR_BayerRG2RGB_EA", ColorConversionCodes::COLOR_BayerRG2RGB_EA)
        .value("COLOR_BayerGR2RGB_EA", ColorConversionCodes::COLOR_BayerGR2RGB_EA)
        .value("COLOR_COLORCVT_MAX", ColorConversionCodes::COLOR_COLORCVT_MAX);

}
