cmake_minimum_required(VERSION 3.0)
project(volumeARAP)
set(CMAKE_CXX_STANDARD 11)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/Eigen/cmake")

#Eigen3
find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Eigen)

#OpenVolumeMesh
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../OpenVolumeMesh/src)
link_directories(${PROJECT_SOURCE_DIR}/../OpenVolumeMesh/build/Build/lib)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)

#aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src ${hello_src})

add_definitions(
      -D_USE_MATH_DEFINES -DNOMINMAX -DDUSE_OPENMP -D_SCL_SECURE_NO_DEPRECATE
  )

#BUILD
SET(HEADERS  
./src/ARAPDeform.h ./src/MyUtils.h
)
add_executable(${PROJECT_NAME} ./src/main.cpp ./src/MyUtils.cpp ./src/yyjARAPDeform.cpp ${HEADERS})
#add_executable(${PROJECT_NAME} ${hello_src})
target_link_libraries(${PROJECT_NAME} OpenVolumeMesh)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG " )