# openvino_2022.1.0.643_MaskRCNN

# CMD
.\openvinoMaskRCNNNewAPI.exe ..\models\mask_rcnn_R_50_FPN_1x.onnx ..\testImages\COCO_val2014_000000001722.jpg
 
# reference
maskrcnn-benchmark：
https://github.com/BowenBao/maskrcnn-benchmark/tree/onnx_stage_mrcnn

onnx inference env：pytorch1.2 onnx1.8.0
training env：pytorch1.6+ and serialize the model

openvino
https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN.html

openmmlab/mmdeploy
https://github.com/open-mmlab/mmdeploy
openvino need compile the latest version according to this https://github.com/openvinotoolkit/openvino/commit/4be0c595059e0eab6166b174f3c9514cbb945d83
