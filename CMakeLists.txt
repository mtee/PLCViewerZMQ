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

IF(PROTOBUF_FOUND)
   MESSAGE(STATUS "Found Protobuf: ${PROTOBUF_INCLUDE_DIR}")
ENDIF()
IF(PROTOBUF_FOUND)
   MESSAGE(STATUS "Found Protobuf: ${PROTOBUF_LIBRARIES}")
ENDIF()
include_directories(${PROTOBUF_INCLUDE_DIR})
#POCO
set(ENV{Poco_DIR} /usr/local/lib)
find_package(Poco REQUIRED Data Net)
include_directories(${Poco_INCLUDE_DIRS})

#Attentive Framework CPP ZMQ implementation
find_package(CommonsCPP REQUIRED)
include_directories(${CommonsCPP_INCLUDE_DIRS})

#FLANN
set(FLANN_INCLUDE_DIR /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/src/cpp)
#set(FLANN_LIBRARY /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib/libflann_cpp.so)
set(FLANN_LIBRARY /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib/libflann_cpp_s.a)
set(FLANN_LIBRARY_DIR /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib)
include_directories (${FLANN_INCLUDE_DIR})
#link_directories("/home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib")

#find_package (LAPACK)
#find_package (GMM)
#find_package (OpenMesh)
#find_package( ANN REQUIRED )

include_directories (
    ${LAPACK_INCLUDE_DIR}
    ${GMM_INCLUDE_DIR}
    ${OPENMESH_INCLUDE_DIR}
    ${ANN_INCLUDE_DIR}
)

link_directories (
  ${LAPACK_LIBRARY_DIR}
  ${OPENMESH_LIBRARY_DIR}
  ${ANN_LIBRARY_DIR}
)


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
   
# ACG

add_executable(zmq_req_rep_app 
    src/zmq_req_rep_app.cpp
    src/semanticgridentry.cpp
    src/pointcloudmapping.cpp
    src/vtkImageMatSource.cpp
)

add_executable(video_app 
    src/video_app.cpp
    src/semanticgridentry.cpp
    src/pointcloudmapping.cpp
    src/vtkImageMatSource.cpp
)

add_executable(arcore_merge_app 
    src/arcore_merge_app.cpp
    src/semanticgridentry.cpp
    src/pointcloudmapping.cpp
    src/vtkImageMatSource.cpp
)

set(LIBS
"/home/mikhail/workspace/ACG-localizer/lib/libacs_localizerd.so"
${PCL_LIBRARIES}
${OpenCV_LIBS}
${Poco_Foundation_LIBRARY}
${Poco_Util_LIBRARY}
${Poco_Data_LIBRARY}
${Poco_Net_LIBRARY}
${ZMQ_LIBRARIES}
${cppzmq_LIBRARY}
${PROTOBUF_LIBRARIES}
${PROTOBUF_LITE_LIBRARIES}
${CommonsCPP_LIBRARIES}
#${OPENMESH_LIBRARY}
#${LAPACK_LIBRARY}
#${LAPACK_LIBRARIES}
#${GMM_LIBRARY}
#${ANN_LIBRARIES}
#${FLANN_LIBRARY}
)

target_include_directories(zmq_req_rep_app PUBLIC /home/mikhail/workspace/ACG-localizer/include)
target_link_libraries(zmq_req_rep_app ${LIBS})

target_include_directories(video_app PUBLIC /home/mikhail/workspace/ACG-localizer/include)
target_link_libraries(video_app ${LIBS})

target_include_directories(arcore_merge_app PUBLIC /home/mikhail/workspace/ACG-localizer/include)
target_link_libraries(arcore_merge_app ${LIBS})