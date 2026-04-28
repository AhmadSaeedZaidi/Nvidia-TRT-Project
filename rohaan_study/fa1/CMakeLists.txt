cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project(fa1 LANGUAGES CXX CUDA)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY "shared")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(attention_fa_lib STATIC
    src/fa1.cu
)

target_include_directories(attention_fa_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(attention_fa_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(fa1_app main.cu)
target_link_libraries(fa1_app PRIVATE attention_fa_lib)
set_target_properties(fa1_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
