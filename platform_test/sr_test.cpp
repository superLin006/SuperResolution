/**
 * 超分辨率测试程序 - 在Android设备上运行DLA模型
 * 
 * 使用Neuron SDK Runtime API加载DLA模型并运行推理
 * 支持通过adb shell在Android设备上运行
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <memory>
#include <algorithm>

// Neuron SDK Runtime API
#include "neuron/api/RuntimeAPI.h"

// 图片处理库（stb_image - 单文件头文件库）
// 需要下载 stb_image.h 和 stb_image_write.h 到当前目录
// 下载地址: https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// 错误处理宏
#define CHECK_RUNTIME(ret) \
    do { \
        if (ret != 0) { \
            std::cerr << "Runtime error: " << ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

/**
 * 加载图片文件
 */
bool load_image(const std::string& path, std::vector<uint8_t>& data, 
                int& width, int& height, int& channels) {
    int w, h, c;
    unsigned char* img_data = stbi_load(path.c_str(), &w, &h, &c, 3);  // 强制RGB
    if (!img_data) {
        std::cerr << "Failed to load image: " << path << std::endl;
        std::cerr << "Error: " << stbi_failure_reason() << std::endl;
        return false;
    }
    
    width = w;
    height = h;
    channels = 3;
    data.resize(w * h * 3);
    std::memcpy(data.data(), img_data, w * h * 3);
    
    stbi_image_free(img_data);
    return true;
}

/**
 * 保存图片文件
 */
bool save_image(const std::string& path, const std::vector<float>& data,
                int width, int height, int channels) {
    // 转换为uint8_t
    std::vector<uint8_t> uint8_data(width * height * channels);
    for (size_t i = 0; i < uint8_data.size(); ++i) {
        float val = std::max(0.0f, std::min(255.0f, data[i] * 255.0f));
        uint8_data[i] = static_cast<uint8_t>(val);
    }
    
    // 保存为PNG
    int success = stbi_write_png(path.c_str(), width, height, channels, 
                                  uint8_data.data(), width * channels);
    if (!success) {
        std::cerr << "Failed to save image: " << path << std::endl;
        return false;
    }
    return true;
}

/**
 * 将uint8图像转换为float32，归一化到[0,1]
 */
std::vector<float> normalize_image(const std::vector<uint8_t>& uint8_data) {
    std::vector<float> float_data(uint8_data.size());
    for (size_t i = 0; i < uint8_data.size(); ++i) {
        float_data[i] = uint8_data[i] / 255.0f;
    }
    return float_data;
}

/**
 * 运行超分辨率推理
 */
bool run_super_resolution(const std::string& dla_model_path,
                         const std::string& input_image_path,
                         const std::string& output_image_path) {
    std::cout << "==========================================" << std::endl;
    std::cout << "超分辨率测试程序" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "DLA模型路径: " << dla_model_path << std::endl;
    std::cout << "输入图片: " << input_image_path << std::endl;
    std::cout << "输出图片: " << output_image_path << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // 1. 加载输入图片
    std::vector<uint8_t> input_uint8;
    int input_width, input_height, input_channels;
    if (!load_image(input_image_path, input_uint8, input_width, input_height, input_channels)) {
        return false;
    }
    std::cout << "输入图片尺寸: " << input_width << "x" << input_height 
              << " (通道数: " << input_channels << ")" << std::endl;
    
    // 转换为float32并归一化
    std::vector<float> input_float = normalize_image(input_uint8);
    
    // 2. 创建Runtime
    void* runtime = nullptr;
    EnvOptions options;
    std::memset(&options, 0, sizeof(options));
    options.deviceKind = kEnvOptHardware;
    options.CPUThreadNum = 1;
    options.suppressInputConversion = false;
    options.suppressOutputConversion = false;
    
    int ret = NeuronRuntime_create_with_options(nullptr, &options, &runtime);
    if (ret != 0) {
        std::cerr << "Failed to create runtime: " << ret << std::endl;
        return false;
    }
    std::cout << "Runtime创建成功" << std::endl;
    
    // 3. 加载DLA模型 - 使用分步加载流程
    std::cout << "打开DLA模型文件: " << dla_model_path << std::endl;
    
    // 3.1 打开网络文件
    ret = NeuronRuntime_openNetworkFromFile(runtime, dla_model_path.c_str());
    if (ret != 0) {
        std::cerr << "Failed to open DLA model: " << ret << std::endl;
        if (ret == 4) {
            std::cerr << "错误代码 4 (BAD_DATA) 表示数据加载失败，可能原因:" << std::endl;
            std::cerr << "  1. 模型文件格式不正确或损坏" << std::endl;
            std::cerr << "  2. 模型文件与平台不匹配（当前使用 mt6989 SDK）" << std::endl;
            std::cerr << "  3. 模型文件路径错误或无法访问" << std::endl;
            std::cerr << "  4. 模型文件可能不是有效的 DLA 格式" << std::endl;
            std::cerr << "请检查模型文件是否正确转换为 DLA 格式" << std::endl;
        }
        NeuronRuntime_release(runtime);
        return false;
    }
    std::cout << "DLA模型文件已打开" << std::endl;
    
    // 3.2 获取临时缓冲区大小
    size_t temp_size = 0;
    ret = NeuronRuntime_getTempSize(runtime, &temp_size);
    if (ret != 0) {
        std::cerr << "Failed to get temp size: " << ret << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    std::cout << "临时缓冲区大小: " << temp_size << " 字节" << std::endl;
    
    // 3.3 设置临时缓冲区（如果需要）
    if (temp_size > 0) {
        std::vector<uint8_t> temp_buffer(temp_size);
        BufferAttribute temp_attr = {NON_ION_FD};
        ret = NeuronRuntime_setTemp(runtime, temp_size, temp_attr, 0);
        if (ret != 0) {
            std::cerr << "Failed to set temp buffer: " << ret << std::endl;
            NeuronRuntime_release(runtime);
            return false;
        }
        std::cout << "临时缓冲区已设置" << std::endl;
    }
    
    // 3.4 初始化网络
    ret = NeuronRuntime_initializeNetwork(runtime);
    if (ret != 0) {
        std::cerr << "Failed to initialize network: " << ret << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    std::cout << "DLA模型加载成功" << std::endl;
    
    // 4. 获取输入输出信息
    size_t input_count, output_count;
    ret = NeuronRuntime_getInputNumber(runtime, &input_count);
    CHECK_RUNTIME(ret);
    ret = NeuronRuntime_getOutputNumber(runtime, &output_count);
    CHECK_RUNTIME(ret);
    
    std::cout << "输入数量: " << input_count << std::endl;
    std::cout << "输出数量: " << output_count << std::endl;
    
    // 获取输入尺寸（假设只有一个输入）
    size_t input_size;
    ret = NeuronRuntime_getSingleInputSize(runtime, &input_size);
    if (ret != 0) {
        // 如果有多个输入，需要逐个获取
        for (uint64_t handle = 0; handle < input_count; ++handle) {
            size_t size;
            ret = NeuronRuntime_getInputSize(runtime, handle, &size);
            CHECK_RUNTIME(ret);
            std::cout << "输入 " << handle << " 大小: " << size << " 字节" << std::endl;
            if (handle == 0) input_size = size;
        }
    } else {
        std::cout << "输入大小: " << input_size << " 字节" << std::endl;
    }
    
    // 获取输出尺寸
    size_t output_size;
    ret = NeuronRuntime_getSingleOutputSize(runtime, &output_size);
    if (ret != 0) {
        for (uint64_t handle = 0; handle < output_count; ++handle) {
            size_t size;
            ret = NeuronRuntime_getOutputSize(runtime, handle, &size);
            CHECK_RUNTIME(ret);
            std::cout << "输出 " << handle << " 大小: " << size << " 字节" << std::endl;
            if (handle == 0) output_size = size;
        }
    } else {
        std::cout << "输出大小: " << output_size << " 字节" << std::endl;
    }
    
    // 5. 准备输入缓冲区
    // 模型期望输入: [1, H, W, 3] float32, 归一化到[0,1]
    size_t expected_input_size = input_width * input_height * input_channels * sizeof(float);
    if (input_size != expected_input_size) {
        std::cerr << "输入尺寸不匹配! 期望: " << expected_input_size 
                  << " 字节, 模型要求: " << input_size << " 字节" << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    
    // 6. 准备输出缓冲区
    std::vector<float> output_data(output_size / sizeof(float));
    
    // 7. 设置输入输出缓冲区
    BufferAttribute input_attr = {NON_ION_FD};
    BufferAttribute output_attr = {NON_ION_FD};
    
    ret = NeuronRuntime_setSingleInput(runtime, input_float.data(), input_size, input_attr);
    if (ret != 0) {
        std::cerr << "Failed to set input buffer: " << ret << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    
    ret = NeuronRuntime_setSingleOutput(runtime, output_data.data(), output_size, output_attr);
    if (ret != 0) {
        std::cerr << "Failed to set output buffer: " << ret << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    
    // 8. 运行推理
    std::cout << "\n开始推理..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ret = NeuronRuntime_inference(runtime);
    if (ret != 0) {
        std::cerr << "推理失败: " << ret << std::endl;
        NeuronRuntime_release(runtime);
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    std::cout << "推理完成，耗时: " << duration << " ms" << std::endl;
    std::cout << "FPS: " << (1000.0f / duration) << std::endl;
    
    // 9. 计算输出尺寸
    // 从输出缓冲区大小推断输出尺寸: output_size = width * height * channels * sizeof(float)
    int output_channels = 3;
    size_t output_pixels = output_size / (sizeof(float) * output_channels);
    // 假设输出是输入的整数倍（通常为2x或4x）
    int scale_factor = 2;  // 默认2x，可以根据模型配置调整
    int output_width = input_width * scale_factor;
    int output_height = output_pixels / output_width;
    
    // 如果计算出的高度不合理，使用scale_factor
    if (output_height <= 0 || output_height > input_height * 4) {
        output_width = input_width * scale_factor;
        output_height = input_height * scale_factor;
    }
    
    std::cout << "输出尺寸: " << output_width << "x" << output_height 
              << " (通道数: " << output_channels << ")" << std::endl;
    
    // 10. 保存结果
    if (!save_image(output_image_path, output_data, output_width, output_height, output_channels)) {
        NeuronRuntime_release(runtime);
        return false;
    }
    std::cout << "\n结果已保存到: " << output_image_path << std::endl;
    
    // 11. 释放资源
    NeuronRuntime_release(runtime);
    
    std::cout << "\n✅ 测试完成！" << std::endl;
    return true;
}

/**
 * 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "用法: " << program_name << " [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <路径>        DLA模型文件路径 (必需)" << std::endl;
    std::cout << "  --input <路径>        输入图片路径 (必需)" << std::endl;
    std::cout << "  --output <路径>       输出图片路径 (必需)" << std::endl;
    std::cout << "  --help, -h            显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name 
              << " --model /data/local/tmp/model_mt6989.dla"
              << " --input /data/local/tmp/test.png"
              << " --output /data/local/tmp/output.png" << std::endl;
}

/**
 * 主函数
 */
int main(int argc, char* argv[]) {
    std::string dla_model_path;
    std::string input_image_path;
    std::string output_image_path;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            dla_model_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_image_path = argv[++i];
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // 检查必需参数
    if (dla_model_path.empty() || input_image_path.empty() || output_image_path.empty()) {
        std::cerr << "错误: 缺少必需参数" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // 运行测试
    if (!run_super_resolution(dla_model_path, input_image_path, output_image_path)) {
        return 1;
    }
    
    return 0;
}
