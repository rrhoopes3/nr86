cmake_minimum_required(VERSION 3.24)
project(nr86_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nr86_capture MODULE nr86_capture.cpp)

set(RESHADE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/reshade-sdk/include")
target_include_directories(nr86_capture PRIVATE "${RESHADE_INCLUDE}")

target_compile_definitions(nr86_capture PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(_addon_suffix ".addon64")
else()
	set(_addon_suffix ".addon32")
endif()
set_target_properties(nr86_capture PROPERTIES
	OUTPUT_NAME "nr86_capture"
	SUFFIX "${_addon_suffix}"
	PREFIX ""
)

if(MSVC)
	target_compile_options(nr86_capture PRIVATE /W3 /EHsc)
endif()
