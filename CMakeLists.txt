cmake_minimum_required(VERSION 3.16)
project(Lab2)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)

add_executable(Lab2 main.cpp)
TARGET_LINK_LIBRARIES(Lab2 PRIVATE OpenMP::OpenMP_CXX)
