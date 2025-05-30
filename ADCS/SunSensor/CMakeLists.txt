cmake_minimum_required(VERSION 3.10)
project(SunSensor)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(sun_sensor sun_sensor_cpp.cpp)

find_package(Eigen3 REQUIRED)
target_include_directories(sun_sensor PRIVATE ${EIGEN3_INCLUDE_DIR})