#include <iostream>
//#include "openvinoSeg.h"
#include "openvinoMaskRCNNMMlab.h"
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace std::literals;

cv::RNG rng(12345);

int main(int argc, char** argv) {

    auto detector = openvinoMaskRCNNMMlab("..\\models\\end2end_cpu.onnx");
    float *pred_bbox;
    int64 *pred_label;
    float *pred_mask;
    int count;
    cv::Mat img = cv::imread("..\\testImages\\test.jpg");

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();

    detector.detect(img, pred_label, pred_bbox, pred_mask, count);
    std::cout << "success!!!!" << std::endl;

    const auto end = std::chrono::steady_clock::now();

    std::cout
            << "model detector takes "
            << (end - start) / 1ms << "ms"
            <<std::endl;

    std::cout << "get results num: " << count << std::endl;

    float SCORE_THRESHOLD=0.35;

    offset off=detector.GetOffsetInfo();
    const int MASK_STEP_SIZE = 28*28;
    for(int ind=0;ind<count;ind++){
        float score = *(pred_bbox+ind*5+4);

        if(score < SCORE_THRESHOLD)
            continue;

        float x0=*(pred_bbox+ind*5);
        float y0=*(pred_bbox+ind*5+1);
        float x1=*(pred_bbox+ind*5+2);
        float y1=*(pred_bbox+ind*5+3);

        int x0_t = std::max(int((x0-off.left)/off.scala_factor), 0);
        int y0_t = std::max(int((y0-off.top)/off.scala_factor), 0);
        int x1_t = std::min(int((x1-off.left)/off.scala_factor), img.cols);
        int y1_t = std::min(int((y1-off.top)/off.scala_factor), img.rows);

        cv::rectangle(img, cv::Point(x0_t, y0_t), cv::Point(x1_t, y1_t), cv::Scalar(255,255,0), 2);
        cv::putText(img, std::to_string(*(pred_label+ind)), cv::Point(x0_t, y0_t),
                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1, false);

        float* mask_ind = pred_mask + MASK_STEP_SIZE * ind ;
        cv::Mat r_mat_(28, 28, CV_32FC1, mask_ind);
        cv::Mat r_mat;
        r_mat_.convertTo(r_mat, CV_8UC1);

        cv::Mat resize_back;
        cv::resize(r_mat, resize_back, cv::Size(int(x1_t-x0_t+1),int(y1_t-y0_t+1)));
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(resize_back, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(img, contours, -1, color, 1, 1,cv::noArray(), INT_MAX, cv::Point(x0_t , y0_t));

    }

    cv::imwrite("result.png", img);


    return 0;
}
