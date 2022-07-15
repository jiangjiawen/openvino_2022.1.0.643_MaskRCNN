//
// Created by JiangJiawen on 2022/7/13.
//

#ifndef OPENVINOYOLOV7INSTNEWAPI_OPENVINOYOLOV7INST_H
#define OPENVINOYOLOV7INSTNEWAPI_OPENVINOYOLOV7INST_H

#include <vector>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

#include "slog.h"

using namespace ov::preprocess;

class openvinoYoloV7Inst{
public:
    explicit openvinoYoloV7Inst(const std::string model_path);
    ~openvinoYoloV7Inst();

    void initInputOutputInfo();
    void printInputAndOutputsInfo(const ov::Model& network);

    void preprocessImage(const cv::Mat& oimg);
    void detect(const cv::Mat &image, int64 *&pred_l, float *&pred_s, bool *&pred_m);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    PrePostProcessor *ppp;

    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    std::string input_tensor_name;
    std::vector<std::string> outputs_tensor_name;

    std::shared_ptr<unsigned char> input_data;

    int img_w = 640;
    int img_h = 640;

    int channel_num = 3;
};

#endif //OPENVINOYOLOV7INSTNEWAPI_OPENVINOYOLOV7INST_H
