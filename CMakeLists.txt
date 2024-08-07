cmake_minimum_required(VERSION 3.28)
project (TEST LANGUAGES CXX DESCRIPTION "OpenCV examples")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -mtune=native -march=native")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")
	set(CMAKE_CXX_FLAGS_RelWithDebInfo "${CMAKE_CXX_FLAGS} -O2 -g")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_INCLUDE_DIRS})

add_executable(imread imread.cpp)
target_link_libraries(imread ${OpenCV_LIBS})
add_executable(mysample mysample.cpp jpgheaders.cpp)
target_link_libraries(mysample ${OpenCV_LIBS})

add_executable(test classtest.cpp)
