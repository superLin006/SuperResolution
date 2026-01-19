cmake_minimum_required(VERSION 3.10)
project(sr_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 配置选项
option(USE_ANDROID "Build for Android" OFF)
option(PLATFORM "Target platform" "mt6989")

# 设置Neuron SDK路径（如果未设置，需要用户指定）
if(NOT DEFINED NEURON_SDK_PATH)
    set(NEURON_SDK_PATH "${CMAKE_SOURCE_DIR}/../../neuropilot-sdk")
    message(STATUS "NEURON_SDK_PATH not set, using default: ${NEURON_SDK_PATH}")
endif()

# 检查Neuron SDK路径
if(NOT EXISTS "${NEURON_SDK_PATH}")
    message(FATAL_ERROR "Neuron SDK not found at ${NEURON_SDK_PATH}. Please set NEURON_SDK_PATH")
endif()

# 设置平台相关的路径
set(NEURON_INCLUDE_DIR "${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/include")
set(NEURON_LIB_DIR "${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib")

# 检查头文件目录
if(NOT EXISTS "${NEURON_INCLUDE_DIR}")
    message(FATAL_ERROR "Neuron SDK include directory not found: ${NEURON_INCLUDE_DIR}")
endif()

# 包含目录
include_directories(
    ${NEURON_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接目录
link_directories(${NEURON_LIB_DIR})

# 源文件
set(SOURCES
    sr_test.cpp
)

# 创建可执行文件
add_executable(sr_test ${SOURCES})

# 检查库文件架构
if(EXISTS "${NEURON_LIB_DIR}/libneuron_runtime.so")
    # 尝试检查库文件架构（需要readelf工具）
    execute_process(
        COMMAND readelf -h "${NEURON_LIB_DIR}/libneuron_runtime.so.8.2.30" 2>/dev/null
        COMMAND grep -E "Machine|Class" 2>/dev/null
        OUTPUT_VARIABLE LIB_ARCH
        ERROR_QUIET
    )
    
    if(LIB_ARCH MATCHES "ARM|AArch64")
        message(STATUS "检测到ARM架构的Neuron SDK库 - 需要使用Android NDK交叉编译")
        if(NOT ANDROID_NDK)
            message(WARNING "未设置ANDROID_NDK，链接可能会失败")
            message(WARNING "请使用Android NDK进行交叉编译，或使用build.sh脚本")
        endif()
    endif()
endif()

# Android特定设置
if(USE_ANDROID OR ANDROID)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(sr_test
        neuron_runtime
        ${log-lib}
        ${android-lib}
    )
else()
    # 非Android平台（如Linux）- 注意：这通常不会成功，因为库是ARM架构的
    message(WARNING "在非Android平台编译 - Neuron SDK库是为ARM架构编译的")
    message(WARNING "建议使用Android NDK进行交叉编译")
    message(WARNING "或者使用build.sh脚本，它会自动处理交叉编译")
    
    target_link_libraries(sr_test
        neuron_runtime
        pthread
        dl
    )
endif()

# 编译选项
target_compile_options(sr_test PRIVATE
    -Wall
    -Wextra
)

# 输出信息
message(STATUS "Building for platform: ${PLATFORM}")
message(STATUS "Neuron SDK path: ${NEURON_SDK_PATH}")
message(STATUS "Include directory: ${NEURON_INCLUDE_DIR}")
message(STATUS "Library directory: ${NEURON_LIB_DIR}")
