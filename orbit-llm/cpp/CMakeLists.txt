cmake_minimum_required(VERSION 3.23)

project(orbit_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO comes from the pip package; GenAI comes from the local source build.
set(OPENVINO_GENAI_BUILD_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../openvino.genai/build"
    CACHE PATH "OpenVINO GenAI build directory")

find_package(OpenVINOGenAI REQUIRED
    PATHS "${OPENVINO_GENAI_BUILD_DIR}"
    NO_DEFAULT_PATH)

add_executable(orbit_run orbit_run.cpp telemetry.hpp)
target_link_libraries(orbit_run PRIVATE openvino::genai psapi pdh advapi32)

# Interactive local chat application (the deployment).
add_executable(orbit_chat orbit_chat.cpp telemetry.hpp)
target_link_libraries(orbit_chat PRIVATE openvino::genai psapi)

if(MSVC)
    target_compile_options(orbit_run PRIVATE /W4 /utf-8)
    target_compile_options(orbit_chat PRIVATE /W4 /utf-8)
endif()

# Put the GenAI and tokenizers DLLs next to the executable.
# openvino_genai.dll loads openvino_tokenizers.dll from its own folder,
# so both must live together. OpenVINO core DLLs are found through PATH.
add_custom_command(TARGET orbit_run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENVINO_GENAI_BUILD_DIR}/openvino_genai/openvino_genai.dll"
        "${OPENVINO_GENAI_BUILD_DIR}/openvino_genai/openvino_tokenizers.dll"
        "$<TARGET_FILE_DIR:orbit_run>")
