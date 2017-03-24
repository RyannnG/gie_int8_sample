cmake_minimum_required (VERSION 2.8)
project (MakeData)

set(CMAKE_CXX_FLAGS -std=c++11)
aux_source_directory(. SRCS)

add_executable(MakeData ${SRCS})

set(EXTRA_LIBS m pthread opencv_imgproc opencv_core opencv_highgui)
target_link_libraries (MakeData ${EXTRA_LIBS})
