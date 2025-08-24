cmake_minimum_required(VERSION 3.15)
project(SodyoAssignment LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenCV via vcpkg
find_package(OpenCV REQUIRED)

# קובץ המקור
add_executable(SodyoAssignment "Sodyo assigment.cpp")

# include + link
target_include_directories(SodyoAssignment PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SodyoAssignment PRIVATE ${OpenCV_LIBS})
