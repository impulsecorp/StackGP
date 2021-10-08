cmake_minimum_required(VERSION 3.3)
project(fastgp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native")

find_package(PythonLibs REQUIRED)
include(FindPythonLibs)

set(Boost_USE_STATIC_LIBS       OFF) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS
        date_time
	program_options
        system
        filesystem
        python36
        numpy36
        serialization
        thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-DUSE_BOOST_PYTHON)
add_definitions(-DUSE_BOOST_RANDOM)
add_definitions(-DVDEBUG)

add_executable(main ${PROJECT_SOURCE_DIR} main.cpp random.cpp)

target_link_libraries(main ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} boost_thread boost_program_options)
