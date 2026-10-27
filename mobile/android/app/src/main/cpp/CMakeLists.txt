cmake_minimum_required(VERSION 3.22)
project(univoice_whisper C CXX)

set(CMAKE_CXX_STANDARD 17)

# whisper.cpp from the git submodule (mobile/android/third_party/whisper.cpp),
# passed in by app/build.gradle.kts.
if(NOT WHISPER_DIR)
    message(FATAL_ERROR "WHISPER_DIR is not set")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# Cross-compiling: never tune for the build machine's CPU.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# ggml's own thread pool; avoids shipping the NDK's OpenMP runtime.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Baseline armv8-a runs on every 64-bit phone. Phones with Cortex-A55/A75 or
# newer are faster with -ParmArch=armv8.2-a+dotprod (see README), but that
# build crashes on older armv8.0 cores.
if(UNIVOICE_ARM_ARCH)
    set(GGML_CPU_ARM_ARCH ${UNIVOICE_ARM_ARCH} CACHE STRING "" FORCE)
endif()

add_subdirectory(${WHISPER_DIR} whisper-build)

add_library(univoice_whisper SHARED whisper_jni.cpp)
target_link_libraries(univoice_whisper PRIVATE whisper log)
