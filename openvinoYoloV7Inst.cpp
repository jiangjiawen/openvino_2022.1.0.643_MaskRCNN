//
// Created by JiangJiawen on 2022/7/13.
//
#include "openvinoYoloV7Inst.h"

openvinoYoloV7Inst::openvinoYoloV7Inst(const std::string model_path) {
    model = core.read_model(model_path);
    printInputAndOutputsInfo(*model);
    ppp = new PrePostProcessor(model);
    initInputOutputInfo();
    model = ppp->build();
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
}

openvinoYoloV7Inst::~openvinoYoloV7Inst() {
    delete ppp;
}

void openvinoYoloV7Inst::initInputOutputInfo() {
    input_tensor_name = model->input().get_any_name();

    for(const auto& out:model->outputs()){
        outputs_tensor_name.emplace_back(out.get_any_name());
    }
//    const ov::Shape& out_shape = model->output().get_shape();

    InputInfo& input_info = ppp->input(input_tensor_name);

    const ov::Layout tensor_layout{"NHWC"};
    input_info.tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ColorFormat::BGR)
            .set_layout(tensor_layout)
            .set_spatial_static_shape(img_h, img_w);

    input_info.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ColorFormat::BGR);

    input_info.model().set_layout("NHWC");

//    for(const auto& out_name:outputs_tensor_name){
//        OutputInfo& output_info = ppp->output(out_name);
//        output_info.tensor().set_element_type(ov::element::i32);
//    }

}

void openvinoYoloV7Inst::printInputAndOutputsInfo(const ov::Model& network) {
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

void openvinoYoloV7Inst::preprocessImage(const cv::Mat &oimg) {
    size_t size = img_w * img_h * oimg.channels();
    input_data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    cv::Mat resized(cv::Size(img_w, img_h), oimg.type(), input_data.get());

    cv::resize(oimg, resized, cv::Size(img_w, img_h));
}

void openvinoYoloV7Inst::detect(const cv::Mat &image, int64 *&pred_l, float *&pred_s, bool *&pred_m) {
    preprocessImage(image);

    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {1, static_cast<unsigned long long>(img_h), static_cast<unsigned long long>(img_w), 3};

    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

    infer_request.set_tensor(input_tensor_name, input_tensor);
    infer_request.infer();

    ov::Tensor output_tensor_label = infer_request.get_tensor("labels");
    ov::Tensor output_tensor_score = infer_request.get_tensor("scores");
    ov::Tensor output_tensor_mask = infer_request.get_tensor("masks");

    pred_l = output_tensor_label.data<int64>();
    pred_s = output_tensor_score.data<float>();
    pred_m = output_tensor_mask.data<bool>();
}

