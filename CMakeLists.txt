cmake_minimum_required(VERSION 4.0)
project(Kalman_Filter_2)

set(CMAKE_CXX_STANDARD 20)

add_executable(Kalman_Filter_2 main.cpp)

add_subdirectory(src)

target_link_libraries(Kalman_Filter_2 kalman)
