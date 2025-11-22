// Modified version: reads one wrench file and uses R from CSV

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "kalman.hpp"


const double mass = 0.932;
const Eigen::RowVector3d rs(0,0,0.0);
const Eigen::Vector3d gw(0,0,0.-9.81); //Gravity vector

const double sigma_k = 0.5;
const double ff = 698.3;    // FTS frequency
const double fr = 100.2;
const double fa = 254.3;

const Eigen::Vector3d sfSigmaF(0.3090,0.1110,1.4084);
const Eigen::Vector3d stSigmaT(0.0069,0.0175,0.0003);
const Eigen::Vector3d saSigmaA(0.4193,0.1387,0.9815);

const double sf = 250;
const double st = 5000;
const double sa = 100;

// Skew symmetric matrix
Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S <<     0, -v.z(),  v.y(),
            v.z(),     0, -v.x(),
            -v.y(),  v.x(),     0;
    return S;
}

Eigen::Matrix3d Rz(double theta) {
    Eigen::Matrix3d R;
    double c = std::cos(theta);
    double s = std::sin(theta);
    R << c, -s, 0,
            s,  c, 0,
            0,  0, 1;
    return R;
}

Eigen::Matrix3d compute_Rws(double q1, double q2) {
    // Fixed sensor mounting orientation (your constant R)
    Eigen::Matrix3d R_offset;
    R_offset << 0, 1, 0,
            0, 0, 1,
            1, 0, 0;

    // Robot joint rotations (planar arm assumption)
    Eigen::Matrix3d R1 = Rz(q1);
    Eigen::Matrix3d R2 = Rz(q2);

    // Full rotation: sensor → world
    return R1 * R2 * R_offset;
}

Eigen::Vector3d compute_zg(double q1, double q2) {
    Eigen::Matrix3d Rws = compute_Rws(q1, q2);
    return Rws.transpose() * gw;
}


struct wrench_data {
    long long time;
    double fx, fy, fz;
    double tx, ty, tz;
};

bool open_csv(const std::string& filename, std::ifstream& file) {
    file.open(filename);
    return file.is_open();
}

std::vector<wrench_data> read_wrench_file(const std::string& filename) {
    std::vector<wrench_data> data;

    std::ifstream file;
    if (!open_csv(filename, file)) {
        std::cerr << "Error opening wrench file: " << filename << "\n";
        return data;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { first_line = false; continue; }

        std::stringstream ss(line);
        wrench_data row;
        std::string cell;

        std::getline(ss, cell, ','); row.time = std::stoll(cell);
        std::getline(ss, cell, ','); row.fx = std::stod(cell);
        std::getline(ss, cell, ','); row.fy = std::stod(cell);
        std::getline(ss, cell, ','); row.fz = std::stod(cell);
        std::getline(ss, cell, ','); row.tx = std::stod(cell);
        std::getline(ss, cell, ','); row.ty = std::stod(cell);
        std::getline(ss, cell, ','); row.tz = std::stod(cell);

        data.push_back(row);
    }

    return data;
}

Eigen::MatrixXd load_R_matrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open R matrix file: " << filename << "\n";
        return Eigen::MatrixXd::Zero(6,6);
    }

    std::string line;
    std::vector<double> values;
    int rows = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        rows++;
    }

    int cols = values.size() / rows;
    Eigen::MatrixXd R(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R(i,j) = values[i * cols + j];

    return R;
}

struct kf_entry { long long time; Eigen::VectorXd x_hat; };

void save_kf_csv(const std::string& filename, const std::vector<kf_entry>& data) {
    std::ofstream file(filename);
    file << "time,ax,ay,az,fx,fy,fz,tx,ty,tz\n";
    for (const auto& entry : data) {
        file << entry.time;
        for (int i = 0; i < entry.x_hat.size(); i++) file << "," << entry.x_hat(i);
        file << "\n";
    }
}

// Tuning factors
double q_scale = 0.000000001;   // process noise scale
double r_scale_gravity = 8;  // gravity measurement
double r_scale_ftf = 0.05;      // FTS measurement
double p_scale = 1.0;


int main() {
    std::string wrench_file = "Data_collection/variances/output_wrench_log_test.csv";
    std::string R_file = "Data_collection/R/R_matrix.csv";

    auto wrench_data_vec = read_wrench_file(wrench_file);
    if (wrench_data_vec.empty()) {
        std::cerr << "No wrench data loaded." << std::endl;
        return 1;
    }

    Eigen::MatrixXd R6 = load_R_matrix(R_file);
    if (R6.rows() != 6 || R6.cols() != 6) {
        std::cerr << "Invalid R matrix shape. Expected 6x6." << std::endl;
        return 1;
    }

    // ===== Kalman Filter setup =====
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(9,9);

    Eigen::MatrixXd B(9,3); B.setZero();
    B.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    B.block<3,3>(3,0) = mass * Eigen::Matrix3d::Identity();
    B.block<3,3>(6,0) = mass * skew(rs);

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(9,9);
    P.block<3,3>(0,0) = 0.1*Eigen::Matrix3d::Identity();  // gravity
    P.block<3,3>(3,3) = 1.0*Eigen::Matrix3d::Identity();  // force
    P.block<3,3>(6,6) = 1.0*Eigen::Matrix3d::Identity();  // torque

// Optional: scale to tune how confident you are in initial state

    P *= p_scale;


    // FTS measurement
    Eigen::MatrixXd Hf = Eigen::MatrixXd::Zero(9,9);
    Hf.block<3,3>(0,0) = Eigen::Matrix3d::Identity();  // gravity
    Hf.block<3,3>(3,3) = Eigen::Matrix3d::Identity();  // forces
    Hf.block<3,3>(6,6) = Eigen::Matrix3d::Identity();  // torques

// Create a 9x9 R matrix
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(9,9);

// Set gravity noise for the first 3x3 block
    Eigen::Vector3d saSigmaA(0.4193,0.1387,0.9815);  // standard deviations for fake IMU
    double sa = 100;  // scaling factor

// Gravity block
    R.block<3,3>(0,0) = (sa * saSigmaA).asDiagonal() * r_scale_gravity;

// FTS forces
    R.block<3,3>(3,3) = R6.block<3,3>(0,0) * r_scale_ftf;

// FTS torques
    R.block<3,3>(6,6) = R6.block<3,3>(3,3) * r_scale_ftf;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(9,9) * q_scale;
    Q.block<3,3>(0,0) *= 10.0; //gravity
    Q.block<3,3>(3,3) *= 12.0; //Forces
    Q.block<3,3>(6,6) *= 0.5; // torques



    kalman_filter KF(0,A,B,Q,R,Hf,P);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(9);

    x0 << 0,0,-9.81,wrench_data_vec[0].fx, wrench_data_vec[0].fy, wrench_data_vec[0].fz,
            wrench_data_vec[0].tx, wrench_data_vec[0].ty, wrench_data_vec[0].tz;

    KF.init(0, x0);

    std::vector<kf_entry> results;

    long long prev_time = wrench_data_vec[0].time;

    for (const auto& w : wrench_data_vec) {
        long long dt = w.time - prev_time;
        prev_time = w.time;

        KF.predict(Eigen::VectorXd::Zero(9));

        // Compute gravity vector in sensor frame
        double q1 = 0.0; // replace with joint angles if available
        double q2 = 0.0;
        Eigen::Vector3d zg = compute_zg(q1, q2);

        // Build measurement vector
        Eigen::VectorXd z(9);
        z << zg(0), zg(1), zg(2),  // gravity acceleration
                w.fx, w.fy, w.fz,     // FTS forces
                w.tx, w.ty, w.tz;     // FTS torques

        KF.update(z, Hf, R);

        results.push_back({w.time, KF.getXHat()});
    }

    save_kf_csv("Data_collection/filtered/Filtered_wrench_output.csv", results);
    std::cout << "Filtered data saved to Filtered_wrench_output.csv\n";

    return 0;
}
