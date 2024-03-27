cmake_minimum_required(VERSION 3.20)



project(rknn_yolov5_demo)
file(GLOB_RECURSE SOURCES "src/*.cc" "deepsort/src/*.cc")
file(GLOB_RECURSE SOURCES "src/*.cpp" "deepsort/src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "deepsort/include/*.h")
file(GLOB_RECURSE HEADERS "include/*.hpp" "deepsort/include/*.hpp")
set(
        include_libs
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/deepsort/include"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(/opt/ros/humble/include/rclcpp/rclcpp)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)
include_directories(${PROJECT_SOURCE_DIR}/deepsort/include)

message("${PROJECT_SOURCE_DIR}/deepsort/include")
# opencv
find_package(OpenCV REQUIRED)

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

include_directories(include ${rclcpp_INCLUDE_DIRS})

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(
        ${include_libs}
        "include"d
        "deepsort/include"
)

add_executable(rknn_yolov5_demo
        ${SOURCES}
        src/main.cc
        src/postprocess.cc
        src/preprocess.cc
        src/rkYolov5s.cc
        src/
)
message("${SOURCES}")
ament_target_dependencies(rknn_yolov5_demo
  "rclcpp"
  "std_msgs"
)

target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
  ${rclcpp_LIBRARIES}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
