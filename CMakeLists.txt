cmake_minimum_required(VERSION 3.0.0)
project(demo VERSION 0.1.0)

Set(ONNXRUNTIME_DIR /home/yp/onnx/onnxruntime-linux-x64-gpu-1.12.1)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")


include(CTest)
include_directories(${PROJECT_SOURCE_DIR}/include)
enable_testing()

find_package(OpenCV REQUIRED)

add_executable(demo ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/Helpers.cpp)

target_link_libraries(demo ${OpenCV_LIBS})
target_include_directories(demo PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_libraries(demo "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
