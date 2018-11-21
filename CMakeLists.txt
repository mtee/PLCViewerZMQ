#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
#Name your project here
project(dsac++)
SET(CommonsCPP_ROOT_DIR "/home/mikhail/workspace/JAMES/AttentiveFrameworkBase/CommonsCPP")
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")
#Sends the -std=c99 flag to the gcc compiler
add_definitions(-std=c++11)
file(GLOB_RECURSE target_for_ide_srcs "*.h" "*.hpp" "*.hxx" "*.c" "*.cpp" "*.cxx")
add_custom_target(headers SOURCES ${target_for_ide_srcs})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

source_group("" FILES ${UTILS_HEADERS})
# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
        message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
        message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS} )



#PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#ZMQ
find_package(ZMQ REQUIRED)
find_package(cppzmq REQUIRED)
include_directories(${ZMQ_INCLUDE_DIRS})

#PROTOBUF
FIND_PACKAGE(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

#Attentive Framework CPP ZMQ implementation
find_package(CommonsCPP REQUIRED)
include_directories(${CommonsCPP_INCLUDE_DIRS})


# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

add_library("opencv_dep_cudart" UNKNOWN IMPORTED)
set_target_properties("opencv_dep_cudart" PROPERTIES IMPORTED_LOCATION /usr/local/cuda/lib64/libcudart.so) 
   
add_executable(main 
    src/main.cpp
    src/pointcloudmapping.cpp
    src/vtkImageMatSource.cpp
)
    

set(LIBS
${PCL_LIBRARIES}
${OpenCV_LIBS}
#${ZMQ_LIBRARIES}
#${cppzmq_LIBRARY}
#${PROTOBUF_LIB}
#${PROTOBUF_LITE_LIB}
#${CommonsCPP_LIBRARIES}
)


target_link_libraries(main ${LIBS})
