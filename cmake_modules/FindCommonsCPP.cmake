# - Find CommonsCPP 
#
# It sets the following variables:
#  CommonsCPP_FOUND         - Set to false, or undefined, if CommonsCPP isn't found.
#  CommonsCPP_INCLUDE_DIRS  - The CommonsCPP include directory.
#  CommonsCPP_LIBRARIES     - The CommonsCPP library to link against.
#
#  Set CommonsCPP_ROOT_DIR environment variable as the path to CommonsCPP root folder.

find_path(CommonsCPP_INCLUDE_DIR NAMES Commons.h PATHS ${CommonsCPP_ROOT_DIR}/build/include)
find_library(CommonsCPP_LIBRARY NAMES libCommonsCPP.so PATHS ${CommonsCPP_ROOT_DIR}/build)

IF (CommonsCPP_INCLUDE_DIR AND CommonsCPP_LIBRARY)
   SET(CommonsCPP_FOUND TRUE)
   SET(CommonsCPP_INCLUDE_DIRS ${CommonsCPP_INCLUDE_DIR} $ENV{CommonsCPP_ROOT_DIR})
   SET(CommonsCPP_LIBRARIES ${CommonsCPP_LIBRARY})
ENDIF (CommonsCPP_INCLUDE_DIR AND CommonsCPP_LIBRARY)

IF (CommonsCPP_FOUND)
   MESSAGE(STATUS "Found CommonsCPP: ${CommonsCPP_LIBRARIES}")
ELSE (CommonsCPP_FOUND)
   MESSAGE(FATAL_ERROR "Could not find CommonsCPP in ${CommonsCPP_ROOT_DIR}/build")
ENDIF (CommonsCPP_FOUND)

