cmake_minimum_required(VERSION 4.0)
project(Kalman_Filter_2)

set(CMAKE_CXX_STANDARD 20)

add_executable(Kalman_Filter_2 main.cpp)
add_executable(Read_Sensor read_sensor.cpp)
add_executable(write_sensor_values write_sensor_values.cpp)
add_executable(Filter Filtered_values.cpp)

add_subdirectory(src)

target_link_libraries(Kalman_Filter_2 kalman)
target_link_libraries(Read_Sensor kalman)
target_link_libraries(write_sensor_values kalman)
target_link_libraries(Filter kalman)