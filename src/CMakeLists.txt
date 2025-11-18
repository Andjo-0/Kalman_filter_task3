add_library(kalman kalman.cpp)
target_include_directories(kalman PUBLIC "${CMAKE_SOURCE_DIR}/include")

include(FetchContent)

FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master   # or "master" but a version tag is better

)

FetchContent_MakeAvailable(Eigen3)


target_link_libraries(kalman Eigen3::Eigen)
