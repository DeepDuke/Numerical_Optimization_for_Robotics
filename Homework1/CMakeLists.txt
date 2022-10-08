cmake_minimum_required(VERSION 3.10.0)
project(Homework1)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonLibs 2.7)
set(PYTHON2.7_INLCUDE_DIRS "/usr/include/python2.7")
include_directories(
    ${PYTHON2.7_INLCUDE_DIRS}
)

add_executable(Homework1 SolveRosenblock.cpp)
target_include_directories(Homework1 PRIVATE ${PYTHON2_INCLUDE_DIRS})
target_link_libraries(Homework1 ${PYTHON_LIBRARIES})
