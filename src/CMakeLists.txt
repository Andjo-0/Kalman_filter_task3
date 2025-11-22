add_library(kalman kalman.cpp MPC.cpp)
target_include_directories(kalman PUBLIC "${CMAKE_SOURCE_DIR}/include")


find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(asio REQUIRED)   # for standalone Asio


target_link_libraries(kalman
        Eigen3::Eigen
        Boost::boost
        Boost::system
        Boost::filesystem
        asio::asio

)