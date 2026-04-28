cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project(naive_fa LANGUAGES CXX CUDA)


if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY "shared") # not using hybrid even tho warnings, since it causes issues with MSVC and CMake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(attention_lib STATIC
    src/k_qk.cu
    src/softmax.cu
    src/pv.cu
)

target_include_directories(attention_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(attention_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(naive_fa_app main.cu)
target_link_libraries(naive_fa_app PRIVATE attention_lib)
set_target_properties(naive_fa_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
