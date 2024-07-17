cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
cmake_policy(VERSION 3.12.0)
project(OpenVolumeMeshExamples)

if (NOT TARGET OpenVolumeMesh::OpenVolumeMesh)
    find_package(OpenVolumeMesh REQUIRED)
endif()

# Add target for first example
add_executable(simple_mesh simple_mesh/simple_mesh.cpp)

target_link_libraries(simple_mesh OpenVolumeMesh::OpenVolumeMesh)

set(OVM_CXX_STANDARD 11 CACHE STRING "C++ standard version to use")
set_property(CACHE OVM_CXX_STANDARD PROPERTY STRINGS 11 14 17 20)

set_target_properties (simple_mesh PROPERTIES
    CXX_STANDARD ${OVM_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

#[[=
if(WIN32)
  # copy exe file to "Build" directory
  # Visual studio will create this file in a subdirectory so we can't use
  # RUNTIME_OUTPUT_DIRECTORY directly here
  add_custom_command (TARGET simple_mesh POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E
                              copy_if_different
                                ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/simple_mesh.exe
                                ${CMAKE_BINARY_DIR}/Examples/simple_mesh.exe)
else()
endif()
=]]
