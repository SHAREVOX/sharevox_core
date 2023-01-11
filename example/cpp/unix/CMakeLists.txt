cmake_minimum_required(VERSION 3.16)

project(SimpleTTS)

add_executable(simple_tts simple_tts.cpp)
set_property(TARGET simple_tts PROPERTY CXX_STANDARD 11)

file(GLOB ONNXRUNTIME_SHARED_LIB ./libonnxruntime.so.* ./libonnxruntime.*.dylib)
target_link_directories(simple_tts PRIVATE ./sharevox_core)


file(GLOB CORE_LIB ./sharevox_core/libsharevox_core.so.* ./sharevox_core/libsharevox_core.*.dylib)
target_link_libraries(simple_tts sharevox_core)
