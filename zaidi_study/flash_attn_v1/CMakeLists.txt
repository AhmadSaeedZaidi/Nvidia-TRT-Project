cmake_minimum_required(VERSION 3.18)
project(FlashAttnV1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

add_executable(flash_attn_v1 
    app/main.cu 
    src/flash_attn.cu
    src/flash_attn_improved.cu
    ../naive_attn/src/k1_v1.cu
    ../naive_attn/src/k1_v2.cu
    ../naive_attn/src/k2_v1.cu
    ../naive_attn/src/k2_v2.cu
    ../naive_attn/src/k3_v1.cu
    ../naive_attn/src/k3_v2.cu
)

target_include_directories(flash_attn_v1 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../naive_attn/include
)

target_compile_options(flash_attn_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --use_fast_math
        -Xcompiler=-Wall,-Wextra
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
    >
)
