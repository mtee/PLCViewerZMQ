#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
#Name your project here
project(dsac++)
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
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#ZMQ
find_package(ZMQ REQUIRED)
find_package(cppzmq REQUIRED)
include_directories(${ZMQ_INCLUDE_DIRS})


#FLANN
set(FLANN_INCLUDE_DIR /home/demouser/acg_localizer/ACG-localizer/flann-1.6.11-src/src/cpp)
set(FLANN_LIBRARY /home/demouser/acg_localizer/ACG-localizer/flann-1.6.11-src/build/lib/libflann_cpp_s.a)
set(FLANN_LIBRARY_DIR /home/demouser/acg_localizer/ACG-localizer/flann-1.6.11-src/build/lib)
include_directories (${FLANN_INCLUDE_DIR})


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
   
# ACG

add_executable(video_app 
    src/video_app.cpp
    src/pointcloudmapping.cpp
    src/vtkImageMatSource.cpp
)

set(LIBS
"/home/demouser/acg_localizer/ACG-localizer/lib/libacs_localizer.so"
${PCL_LIBRARIES}
${OpenCV_LIBS}
${ZMQ_LIBRARIES}
${cppzmq_LIBRARY}
)

target_include_directories(video_app PUBLIC /home/demouser/acg_localizer/ACG-localizer/include)
target_link_libraries(video_app ${LIBS})

#target_include_directories(arcore_merge_app PUBLIC /home/demouser/acg_localizer/ACG-localizer/include)
#target_link_libraries(arcore_merge_app ${LIBS})
