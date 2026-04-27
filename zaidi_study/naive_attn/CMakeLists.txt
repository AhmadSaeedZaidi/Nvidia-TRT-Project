cmake_minimum_required(VERSION 3.18)
project(NaiveAttn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

add_executable(naive_attn 
    app/main.cu 
    src/k1_v1.cu 
    src/k1_v2.cu 
    src/k2_v1.cu 
    src/k2_v2.cu 
    src/k3_v1.cu 
    src/k3_v2.cu
)

target_include_directories(naive_attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add strict compiler warnings and useful CUDA optimization flags
target_compile_options(naive_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        -Xptxas=-v
        --use_fast_math
        -Xcompiler=-Wall,-Wextra,-Wpedantic
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
        -Wpedantic
    >
)
