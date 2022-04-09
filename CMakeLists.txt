cmake_minimum_required(VERSION 3.12)

project(MM)

# aris use c++ 17 standard
set(CMAKE_CXX_STANDARD 17)

# find aris
if(UNIX)
	add_definitions(-DUNIX)
        find_package(aris REQUIRED PATHS /usr/aris)
endif(UNIX)
IF(WIN32)
        find_package(aris REQUIRED PATHS C:/aris/aris-2.0.0.220107)
        find_package (Eigen3 3.3 REQUIRED NO_MODULE)
endif(WIN32)

include_directories(${aris_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})

add_executable(MM src/main.cpp)

if(UNIX)
    target_link_libraries(MM ${aris_LIBRARIES} atomic)
endif(UNIX)
IF(WIN32)
    target_link_libraries(MM ${aris_LIBRARIES} Eigen3::Eigen)
endif(WIN32)

