//
// Created by JiangJiawen on 2022/7/15.
//
#include "openvinoMaskRCNN.h"

openvinoMaskRCNN::openvinoMaskRCNN(const std::string model_path) {
    model = core.read_model(model_path);
    //printInputAndOutputsInfo(*model);
    ppp = new PrePostProcessor(model);
    initInputOutputInfo();
    model = ppp->build();
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
}

openvinoMaskRCNN::~openvinoMaskRCNN() {
    delete ppp;
}

void openvinoMaskRCNN::initInputOutputInfo() {
    input_tensor_name = model->input().get_any_name();

    for(const auto& out:model->outputs()){
        outputs_tensor_name.emplace_back(out.get_any_name());
    }
//    const ov::Shape& out_shape = model->output().get_shape();

    InputInfo& input_info = ppp->input(input_tensor_name);

    const ov::Layout tensor_layout{"HWC"};
    input_info.tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ColorFormat::BGR)
            .set_layout(tensor_layout)
            .set_spatial_static_shape(H, W);

    std::vector<float> mean{102.9801, 115.9465, 122.7717};
//    std::vector<float> mean{122.7717, 102.9801, 115.9465};
    input_info.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ColorFormat::BGR)
            .mean(mean);

    input_info.model().set_layout("CHW");

//    for(const auto& out_name:outputs_tensor_name){
//        OutputInfo& output_info = ppp->output(out_name);
//        output_info.tensor().set_element_type(ov::element::i32);
//    }
}

void openvinoMaskRCNN::printInputAndOutputsInfo(const ov::Model &network) {
    slog::info << "model name: " << network.get_friendly_name() << slog::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        slog::info << "    inputs" << slog::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        slog::info << "        input name: " << name << slog::endl;

        const ov::element::Type type = input.get_element_type();
        slog::info << "        input type: " << type << slog::endl;

        const ov::Shape& shape = input.get_shape();
        slog::info << "        input shape: " << shape << slog::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        slog::info << "    outputs" << slog::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        slog::info << "        output name: " << name << slog::endl;

        const ov::element::Type type = output.get_element_type();
        slog::info << "        output type: " << type << slog::endl;

        const ov::Shape& shape = output.get_shape();
        slog::info << "        output shape: " << shape << slog::endl;
    }
}

void openvinoMaskRCNN::preprocessImage(const cv::Mat &oimg) {
    size_t size = W * H * oimg.channels();
    input_data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    scale_factor = GenerateScale(oimg);
    cv::Mat resized(cv::Size(W, H), oimg.type(), input_data.get());

    int new_shape_w = std::round(oimg.cols * scale_factor);
    int new_shape_h = std::round(oimg.rows * scale_factor);

    float padw = (W - new_shape_w) / 2.;
    float padh = (H - new_shape_h) / 2.;

    top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::Mat middelImg;
    cv::resize(
            oimg, middelImg, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(middelImg,
                       resized,
                       top,
                       bottom,
                       left,
                       right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    //cv::imwrite("resizemakeborder.png", resized);
}

void openvinoMaskRCNN::detect(const cv::Mat &image, int64 *&pred_l, float *&pred_b, float *&pred_s, float *&pred_m, int& count) {
    preprocessImage(image);

    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {static_cast<unsigned long long>(H), static_cast<unsigned long long>(W), 3};

    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

    infer_request.set_tensor(input_tensor_name, input_tensor);
    infer_request.infer();

//    ov::Tensor output_tensor_label = infer_request.get_tensor("6570");
    ov::Tensor output_tensor_label = infer_request.get_tensor("6557");
    count = output_tensor_label.get_size();
    pred_l = output_tensor_label.data<int64>();

//    ov::Tensor output_tensor_bbox = infer_request.get_tensor("6568");
    ov::Tensor output_tensor_bbox = infer_request.get_tensor("6555");
    pred_b = output_tensor_bbox.data<float>();

//    ov::Tensor output_tensor_score = infer_request.get_tensor("6572");
    ov::Tensor output_tensor_score = infer_request.get_tensor("6559");
    pred_s = output_tensor_score.data<float>();

//    ov::Tensor output_tensor_mask = infer_request.get_tensor("6887");
    ov::Tensor output_tensor_mask = infer_request.get_tensor("6866");
//    const ov::Shape& shape = output_tensor_mask.get_shape();
//    slog::info << "        output shape: " << shape << slog::endl;
    pred_m = output_tensor_mask.data<float>();

}

float openvinoMaskRCNN::GenerateScale(const cv::Mat &img) const {
    int origin_w = img.cols;
    int origin_h = img.rows;

    int target_h = H;
    int target_w = W;

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float scale = std::min(ratio_h, ratio_w);
    return scale;
}

offset openvinoMaskRCNN::GetOffsetInfo() {
    offset off{};
    off.top =top;
    off.left =left;
    off.scala_factor=scale_factor;
    return off;
}

