cmake_minimum_required(VERSION 3.5)
project(pocket_tts VERSION 1.0.0 LANGUAGES CXX C)

find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# =============================================================================
# Dependencies
# =============================================================================
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

include(FetchContent)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0 # Or the version you prefer
    EXCLUDE_FROM_ALL
)

# Force yaml-cpp to build as a static library
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(yaml-cpp)

# ONNX Runtime (pre-built) - https://github.com/microsoft/onnxruntime/releases
set(ONNXRUNTIME_VERSION "1.23.2")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(ONNXRUNTIME_ARCH "x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(ONNXRUNTIME_ARCH "aarch64")
    endif()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
endif()

FetchContent_Declare(onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
else()
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

# SentencePiece (static library only) - https://github.com/google/sentencepiece/releases
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)

# Patch SentencePiece for GCC 14 compatibility and CMake deprecation warning
file(WRITE ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake "
file(READ \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" CONTENT)
if(NOT CONTENT MATCHES \"#include <cstdint>\")
    string(REPLACE \"#include <string>\" \"#include <string>\\n#include <cstdint>\" CONTENT \"\${CONTENT}\")
    file(WRITE \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" \"\${CONTENT}\")
    message(STATUS \"Patched sentencepiece_processor.h for GCC 14 compatibility\")
endif()
file(READ \"\${SOURCE_DIR}/CMakeLists.txt\" CMLCONTENT)
string(REGEX REPLACE \"cmake_minimum_required\\(VERSION [0-9.]+\\)\" \"cmake_minimum_required(VERSION 3.14)\" CMLCONTENT \"\${CMLCONTENT}\")
file(WRITE \"\${SOURCE_DIR}/CMakeLists.txt\" \"\${CMLCONTENT}\")
")

FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.1
    GIT_SHALLOW TRUE
    PATCH_COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sentencepiece)
target_compile_options(sentencepiece-static PRIVATE -w)

set(SENTENCEPIECE_INCLUDE_DIRS "${sentencepiece_SOURCE_DIR}/src")

# dr_libs (header-only) - https://github.com/mackron/dr_libs
FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dr_libs)

# =============================================================================
# Targets
# =============================================================================

option(BUILD_SHARED_LIB "Build libpocket_tts.so for FFI (Deno, Python, etc.)" OFF)

set(POCKET_COMMON_INCLUDE_DIRS
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${dr_libs_SOURCE_DIR}
)

set(POCKET_COMMON_LINK_LIBS
    ${ONNXRUNTIME_LIB}
    sentencepiece-static
    yaml-cpp
    nlohmann_json::nlohmann_json
)

function(configure_pocket_target target_name)
    target_include_directories(${target_name} PRIVATE ${POCKET_COMMON_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${POCKET_COMMON_LINK_LIBS})

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(${target_name} PRIVATE pthread dl)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_link_libraries(${target_name} PRIVATE ws2_32)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
                $<TARGET_FILE_DIR:${target_name}>
        )
    else()
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB} $<TARGET_FILE_DIR:${target_name}>
        )
    endif()
endfunction()

# executable
add_executable(ptts ptts.cpp)
configure_pocket_target(ptts)
target_precompile_headers(ptts PRIVATE <nlohmann/json.hpp>)

if (MSVC)
    target_compile_options(ptts PRIVATE /W4 /WX)
else ()
    target_compile_options(ptts PRIVATE -Wall -Wextra -Wpedantic)
endif ()

# Shared library (for FFI from Deno/Python)
# Build with: cmake -B .build -DBUILD_SHARED_LIB=ON
if(BUILD_SHARED_LIB)
    add_library(pocket-tts-lib SHARED ptts_engine.cpp)
    set_target_properties(pocket-tts-lib PROPERTIES
        OUTPUT_NAME "ptts"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    target_compile_definitions(pocket-tts-lib PRIVATE PTT_SHARED_LIB)

    # Link against the unversioned .so directly and override the SONAME dependency
    # so it looks for libonnxruntime.so (what we ship) not libonnxruntime.so.1
    target_link_libraries(pocket-tts-lib PRIVATE
        -l:libonnxruntime.so
        sentencepiece-static
    )
    target_link_directories(pocket-tts-lib PRIVATE ${ONNXRUNTIME_ROOT}/lib)

    target_include_directories(pocket-tts-lib PRIVATE ${POCKET_COMMON_INCLUDE_DIRS})

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(pocket-tts-lib PRIVATE pthread dl)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_link_libraries(pocket-tts-lib PRIVATE ws2_32)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_custom_command(TARGET pocket-tts-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
                $<TARGET_FILE_DIR:pocket-tts-lib>
        )
    else()
        add_custom_command(TARGET pocket-tts-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB} $<TARGET_FILE_DIR:pocket-tts-lib>
        )
        find_program(PATCHELF patchelf)
        if(PATCHELF)
            add_custom_command(TARGET pocket-tts-lib POST_BUILD
                COMMAND ${PATCHELF} --replace-needed libonnxruntime.so.1 libonnxruntime.so $<TARGET_FILE:pocket-tts-lib>
                COMMENT "Patching libonnxruntime SONAME dependency"
            )
        endif()
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== PocketTTS.cpp ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_VERSION}")
message(STATUS "SentencePiece: v0.2.1")
if(BUILD_SHARED_LIB)
    message(STATUS "Targets: ptts, libpocket_tts.so")
else()
    message(STATUS "Targets: ptts")
endif()
message(STATUS "=====================")
message(STATUS "")
