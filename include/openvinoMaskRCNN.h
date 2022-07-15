//
// Created by JiangJiawen on 2022/7/15.
//

#ifndef OPENVINOMASKRCNNNEWAPI_OPENVINOMASKRCNN_H
#define OPENVINOMASKRCNNNEWAPI_OPENVINOMASKRCNN_H

#include <vector>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

#include "slog.h"

using namespace ov::preprocess;

struct offset{
    int top;
    int left;
    float scala_factor;
};

class openvinoMaskRCNN{
public:
    explicit openvinoMaskRCNN(const std::string model_path);
    ~openvinoMaskRCNN();

    void initInputOutputInfo();
    void printInputAndOutputsInfo(const ov::Model& network);

    void preprocessImage(const cv::Mat& oimg);
    void detect(const cv::Mat &image, int64 *&pred_l, float *&pred_b, float *&pred_s, float *&pred_m, int& count);

    float GenerateScale(const cv::Mat& img) const;

    offset GetOffsetInfo();

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    PrePostProcessor *ppp;

    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    std::string input_tensor_name;
    std::vector<std::string> outputs_tensor_name;

    std::shared_ptr<unsigned char> input_data;

    int W = 800;
    int H = 800;

    int channel_num = 3;

    float scale_factor = 0.6;

    int top=0;
    int left=0;
};

#endif //OPENVINOMASKRCNNNEWAPI_OPENVINOMASKRCNN_H
