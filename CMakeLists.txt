cmake_minimum_required(VERSION 3.22)
project(openvinoMaskRCNNNewAPI)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall /Od")
    message("Debug mode:${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    message("Release mode:${CMAKE_CXX_FLAGS_RELEASE}")
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_CXX_FLAGS_RELEASE}")
endif()

include_directories(include)

set(slog include/slog.h slog.cpp)
set(openvinoYoloV7Inst include/openvinoYoloV7Inst.h openvinoYoloV7Inst.cpp)
set(openvinoMaskRCNN include/openvinoMaskRCNN.h openvinoMaskRCNN.cpp)

add_executable(openvinoMaskRCNNNewAPI main.cpp ${slog} ${openvinoYoloV7Inst} ${openvinoMaskRCNN})

set(OpenCV_DIR "C:/Program Files/opencv/opencv451/build")
find_package (OpenCV REQUIRED)
target_link_libraries(openvinoMaskRCNNNewAPI "opencv_core" "opencv_highgui" "opencv_imgcodecs" "opencv_imgproc" "opencv_dnn")

include_directories("${PROJECT_SOURCE_DIR}/3rd_party/openvino/include/ie")
include_directories("${PROJECT_SOURCE_DIR}/3rd_party/openvino/include/ngraph")
include_directories("${PROJECT_SOURCE_DIR}/3rd_party/openvino/include")
set(openvino_lib "${PROJECT_SOURCE_DIR}/3rd_party/openvino/lib/intel64/Release/openvino.lib")
set(openvino_onnx_lib "${PROJECT_SOURCE_DIR}/3rd_party/openvino/lib/intel64/Release/openvino_onnx_frontend.lib")
target_link_libraries(openvinoMaskRCNNNewAPI ${openvino_lib} ${openvino_onnx_lib})
