cmake_minimum_required(VERSION 2.8.4) # Avoid moaning about cygwin
set(CMAKE_CXX_STANDARD 17)

if (DEFINED ENV{VCPKG_ROOT})
  set(VCPKG_CMAKE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
else ()
  set(VCPKG_CMAKE "D:/AI/vcpkg-export-20200114-082720/vcpkg-export-20200114-082720/scripts/buildsystems/vcpkg.cmake") 
endif ()

if (EXISTS ${VCPKG_CMAKE})
  list(APPEND CMAKE_TOOLCHAIN_FILE ${VCPKG_CMAKE})
endif ()

set(name facerec)
project(${name})

find_package(OpenCV REQUIRED) # This must follow the "project" command

add_executable(${name} ${name}.cpp)
target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${name} ${OpenCV_LIBRARIES})

if (WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${name})
endif (WIN32)


