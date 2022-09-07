#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include "Helpers.cpp"

using namespace cv;
using namespace std;


int main(int, char**) {

    Ort::Env env;
    Ort::Session session(nullptr);
    Ort::SessionOptions sessionOptions{nullptr};

    constexpr int64_t numChannles = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannles * height * width;

    const string imgFile = "/home/yp/workDir/cmake_demo/assets/dog.png";
    const string labelFile = "/home/yp/workDir/cmake_demo/assets/imagenet_classes.txt";
    auto modelPath = "/home/yp/workDir/cmake_demo/assets/resnet50v2.onnx";
    const bool isGPU = true;

    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }


    //load labels
    vector<string> labels = loadLabels(labelFile);
    if (labels.empty()){
        cout<< "Failed to load labels: " << labelFile << endl;
        return 1;
    }

    //load image
    const vector<float> imageVec = loadImage(imgFile);
    if (imageVec.empty()){
        cout << "Invalid image format. Must be 224*224 RGB image. " << endl;
        return 1;
    }

    // create session
    cout << "create session. " << endl;
    session = Ort::Session(env, modelPath, sessionOptions);

    // get the number of input
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // define shape
    const array<int64_t, 4> inputShape = {1, numChannles, height, width};
    const array<int64_t, 2> outputShape = {1, numClasses};

    // define array
    array<float, numInputElements> input;
    array<float, numClasses> results;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    copy(imageVec.begin(), imageVec.end(), input.begin());

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    inputNames.push_back(session.GetInputName(0, ort_alloc));
    outputNames.push_back(session.GetOutputName(0, ort_alloc));

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    // run inference
    cout << "run inference. " << endl;
    try{
        for (size_t i = 0; i < 10; i++)
        {
            auto start = std::chrono::system_clock::now();
            session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        }
    }
    catch(Ort::Exception& e) {
        cout << e.what() << endl;
        return 1;
    }

    // sort results
    vector<pair<size_t, float>> indexValuePairs;
    for(size_t i = 0; i < results.size(); ++i){
        indexValuePairs.emplace_back(i, results[i]);
    }
    sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second;});


    // show Top5
    for (size_t i = 0; i < 5; ++i) {
        const auto& result = indexValuePairs[i];
        cout << i + 1 << ": " << labels[result.first] << " " << result.second << endl;
    }
}
