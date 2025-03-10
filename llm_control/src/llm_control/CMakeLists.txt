cmake_minimum_required(VERSION 3.0.2)
project(llm_control)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(llm_control src/llm_control.cpp)
target_link_libraries(llm_control ${catkin_LIBRARIES})