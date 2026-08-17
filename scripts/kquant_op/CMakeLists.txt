cmake_minimum_required(VERSION 3.15)
project(rwkv_kquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

find_package(OpenVINO REQUIRED)

add_library(rwkv_kquant SHARED rwkv_kquant.cpp)
target_link_libraries(rwkv_kquant PRIVATE openvino::runtime)
target_compile_options(rwkv_kquant PRIVATE /O2 /arch:AVX2 /fp:fast /utf-8)
