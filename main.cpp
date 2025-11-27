#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <string>
#include <algorithm>
#include <array>

#include "kalman.hpp"

const double mass = 0.932;
const Eigen::Vector3d rs(0,0,0.044);

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

//Eigen::Vector3d get_deltaGS(const std::string& filename) {}

long long find_earliest_timestamp(const std::string& filename){
    double t = 0;
    int targetRow = 2;
    int targetCol = 1;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file:"<<filename<<std::endl;
        return 0.0;
    }

    std::string line;
    int currentRow = 0;

    while (std::getline(file, line)) {
        currentRow++;
        if (currentRow == targetRow) {
            std::stringstream ss(line);
            std::string cell;
            int currentCol = 0;
            while (std::getline(ss, cell, ',')) {
                currentCol++;
                if (currentCol == targetCol) {
                    try {
                        return std::stod(cell);
                    } catch (...) {
                        std::cerr << "Invalid number format\n";
                        return 0.0;
                    }
                }
            }
        }
    }

    std::cerr << "Value not found\n";
    return 0.0;
}

std::vector<long long> get_shortest_timestamps(const std::vector<std::string>& tags){
    std::vector<long long> shortest;

    for (const std::string& tag : tags) {
        std::vector<long long> times = {
                find_earliest_timestamp(tag + "_accel.csv"),
                find_earliest_timestamp(tag + "_orientations.csv"),
                find_earliest_timestamp(tag + "_wrench.csv")
        };

        long long min_time = *std::min_element(times.begin(), times.end());
        shortest.push_back(min_time);
    }
    return shortest;
}


// ================== Structs ==================

struct accel_data {
    long long time;
    double ax;
    double ay;
    double az;
};

struct orientation_data {
    long long time;
    double r11, r12, r13;
    double r21, r22, r23;
    double r31, r32, r33;
};

struct wrench_data {
    long long time;
    double fx, fy, fz;
    double tx, ty, tz;
};

// ================== Helper ==================

inline bool open_csv(const std::string& filename, std::ifstream& file) {
    file.open(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return false;
    }
    return true;
}

// ================== ACCEL ==================

std::vector<accel_data> read_single_accel_file(const std::string& filename, long long shortest_timestamp) {
    std::vector<accel_data> data;
    std::ifstream file;
    if (!open_csv(filename, file)) return data;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { first_line = false; continue; }

        std::stringstream ss(line);
        accel_data row;
        std::string cell;

        std::getline(ss, cell, ','); row.time = std::stoll(cell) - shortest_timestamp;
        std::getline(ss, cell, ','); row.ax = std::stod(cell);
        std::getline(ss, cell, ','); row.ay = std::stod(cell);
        std::getline(ss, cell, ','); row.az = std::stod(cell);

        data.push_back(row);
    }

    return data;
}

std::vector<std::vector<accel_data>> read_accel_data(
        const std::vector<std::string>& tags,
        const std::vector<long long>& shortest_timestamps)
{
    std::vector<std::vector<accel_data>> all_data;

    for (size_t i = 0; i < tags.size(); ++i) {
        std::string filename = tags[i] + "_accel.csv";
        all_data.push_back(read_single_accel_file(filename, shortest_timestamps[i]));
    }

    return all_data;
}

// ================== ORIENTATION ==================

std::vector<orientation_data> read_single_orientation_file(const std::string& filename, long long shortest_timestamp) {
    std::vector<orientation_data> data;
    std::ifstream file;
    if (!open_csv(filename, file)) return data;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { first_line = false; continue; }

        std::stringstream ss(line);
        orientation_data row;
        std::string cell;

        std::getline(ss, cell, ','); row.time = std::stoll(cell) - shortest_timestamp;
        std::getline(ss, cell, ','); row.r11 = std::stod(cell);
        std::getline(ss, cell, ','); row.r12 = std::stod(cell);
        std::getline(ss, cell, ','); row.r13 = std::stod(cell);
        std::getline(ss, cell, ','); row.r21 = std::stod(cell);
        std::getline(ss, cell, ','); row.r22 = std::stod(cell);
        std::getline(ss, cell, ','); row.r23 = std::stod(cell);
        std::getline(ss, cell, ','); row.r31 = std::stod(cell);
        std::getline(ss, cell, ','); row.r32 = std::stod(cell);
        std::getline(ss, cell, ','); row.r33 = std::stod(cell);

        data.push_back(row);
    }

    return data;
}

std::vector<std::vector<orientation_data>> read_orientation_data(
        const std::vector<std::string>& tags,
        const std::vector<long long>& shortest_timestamps)
{
    std::vector<std::vector<orientation_data>> all_data;

    for (size_t i = 0; i < tags.size(); ++i) {
        std::string filename = tags[i] + "_orientations.csv";
        all_data.push_back(read_single_orientation_file(filename, shortest_timestamps[i]));
    }

    return all_data;
}

// ================== WRENCH ==================

std::vector<wrench_data> read_single_wrench_file(const std::string& filename, long long shortest_timestamp) {
    std::vector<wrench_data> data;
    std::ifstream file;
    if (!open_csv(filename, file)) return data;

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) { first_line = false; continue; }

        std::stringstream ss(line);
        wrench_data row;
        std::string cell;

        std::getline(ss, cell, ','); row.time = std::stoll(cell) - shortest_timestamp;
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

std::vector<std::vector<wrench_data>> read_wrench_data(
        const std::vector<std::string>& tags,
        const std::vector<long long>& shortest_timestamps)
{
    std::vector<std::vector<wrench_data>> all_data;

    for (size_t i = 0; i < tags.size(); ++i) {
        std::string filename = tags[i] + "_wrench.csv";
        all_data.push_back(read_single_wrench_file(filename, shortest_timestamps[i]));
    }

    return all_data;
}



std::array<double, 3> R_T_times_g(const orientation_data& R_data) {
    // Create Eigen matrix from orientation_data
    Eigen::Matrix3d R;
    R << R_data.r11, R_data.r12, R_data.r13,
            R_data.r21, R_data.r22, R_data.r23,
            R_data.r31, R_data.r32, R_data.r33;

    // Map g_w to an Eigen vector
    Eigen::Vector3d g_vec(0.0,0.0, -9.81);

    // Compute R^T * g_w
    Eigen::Vector3d g_s_vec = R.transpose() * g_vec;


    // Convert result back to std::array
    return { g_s_vec[0], g_s_vec[1], g_s_vec[2] };
}

// Compute u_k
Eigen::Vector3d compute_uk(
        const orientation_data& R_prev,
        const orientation_data& R_curr,
        double f_r, double f_f, double f_a)
{
    auto g_prev = R_T_times_g(R_prev);
    auto g_curr = R_T_times_g(R_curr);


    Eigen::Vector3d g_prev_s(Eigen::Vector3d(g_prev[0], g_prev[1], g_prev[2]));
    Eigen::Vector3d g_curr_s(Eigen::Vector3d(g_curr[0], g_curr[1], g_curr[2]));

    Eigen::Vector3d delta_g = (g_curr_s - g_prev_s) * (f_r / (f_f + f_a));

    return delta_g;
}

template<typename T>
void save_accel_csv(const std::string& filename, const std::vector<T>& data) {
    std::ofstream file(filename);
    file << "time,ax,ay,az\n";
    for (const auto& row : data) {
        file << row.time << "," << row.ax << "," << row.ay << "," << row.az << "\n";
    }
}

template<typename T>
void save_orientation_csv(const std::string& filename, const std::vector<T>& data) {
    std::ofstream file(filename);
    file << "time,r11,r12,r13,r21,r22,r23,r31,r32,r33\n";
    for (const auto& row : data) {
        file << row.time << "," << row.r11 << "," << row.r12 << "," << row.r13 << ","
             << row.r21 << "," << row.r22 << "," << row.r23 << ","
             << row.r31 << "," << row.r32 << "," << row.r33 << "\n";
    }
}

template<typename T>
void save_wrench_csv(const std::string& filename, const std::vector<T>& data) {
    std::ofstream file(filename);
    file << "time,fx,fy,fz,tx,ty,tz\n";
    for (const auto& row : data) {
        file << row.time << "," << row.fx << "," << row.fy << "," << row.fz << ","
             << row.tx << "," << row.ty << "," << row.tz << "\n";
    }
}

struct kf_entry { long long time; Eigen::VectorXd x_hat; Eigen::Vector3d compensated_force;Eigen::Vector3d compensated_torque;};

void save_kf_csv(const std::string& filename, const std::vector<kf_entry>& kf_data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Header: time, 9 state values, 6 compensated outputs
    file << "time,a_x,a_y,a_z,F_x,F_y,F_z,T_x,T_y,T_z,z_1,z_2,z_3,z_4,z_5,z_6\n";

    for (const auto& entry : kf_data) {
        file << entry.time; // time first

        // write 9 state values (x_hat is 9x1)
        const Eigen::VectorXd& x = entry.x_hat;
        for (int i = 0; i < x.size(); ++i) {
            file << "," << x(i);
        }

        // compensated_force is Eigen::Vector3d
        const Eigen::Vector3d& Zf = entry.compensated_force;
        for (int i = 0; i < 3; ++i) {
            file << "," << Zf(i);
        }

        // compensated_torque is Eigen::Vector3d
        const Eigen::Vector3d& Zt = entry.compensated_torque;
        for (int i = 0; i < 3; ++i) {
            file << "," << Zt(i);
        }

        file << "\n";
    }
}



int main() {

    // Ensure output folder exists
    std::filesystem::create_directory("Filtered_data");

    // ===== Kalman Filter setup =====
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(9,9);
    Eigen::MatrixXd B(9,3); B.setZero();
    B.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    B.block<3,3>(3,0) = mass * Eigen::Matrix3d::Identity();
    B.block<3,3>(6,0) = mass * skew(rs);

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(9,9);
    P.block<3,3>(0,0) = 0.1*Eigen::Matrix3d::Identity();
    P.block<3,3>(3,3) = 1.0*Eigen::Matrix3d::Identity();

    // FTS measurement
    Eigen::MatrixXd Hf = Eigen::MatrixXd::Zero(6,9);
    Hf.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    Hf.block<3,3>(3,6) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd Rf = Eigen::MatrixXd::Zero(6,6);
    Rf.block<3,3>(0,0) = (sf * sfSigmaF).asDiagonal();
    Rf.block<3,3>(3,3) = (st * stSigmaT).asDiagonal();

    // IMU measurement
    Eigen::MatrixXd Ha = Eigen::MatrixXd::Zero(3,9);
    Ha.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d Ra = (sa * saSigmaA).asDiagonal();

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(9,9);

    kalman_filter KF(0,A,B,Q,Rf,Hf,P);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(9);
    KF.init(0, x0);

    // Rotation matrix from IMU to FTS frame
    Eigen::Matrix3d R_fa;
    R_fa << 0, -1, 0,
            0,  0, 1,
            -1,  0, 0;

    // ===== Load data =====
    std::vector<std::string> tags = { "Data/1-baseline", "Data/2-vibrations", "Data/3-vibrations-contact" };
    std::vector<long long> shortest = get_shortest_timestamps(tags);

    auto accel_sets = read_accel_data(tags, shortest);
    auto orient_sets = read_orientation_data(tags, shortest);
    auto wrench_sets = read_wrench_data(tags, shortest);

    // ===== Save raw data =====
    for (size_t i = 0; i < tags.size(); ++i) {
        std::string tag_name = "Filtered_data/" + std::filesystem::path(tags[i]).filename().string();
        save_accel_csv(tag_name + "_accel_raw.csv", accel_sets[i]);
        save_orientation_csv(tag_name + "_orientations_raw.csv", orient_sets[i]);
        save_wrench_csv(tag_name + "_wrench_raw.csv", wrench_sets[i]);
    }

    // ===== Run Kalman Filter =====

    std::vector<std::vector<kf_entry>> kf_estimates(tags.size());

    Eigen::Matrix<double, 6, 9> Zbc;

// Top row:  [ -mb*I   ,   I    ,   0 ]
    Zbc.block<3,3>(0,0) = -mass * Eigen::Matrix3d::Identity()*-9.81;
    Zbc.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    Zbc.block<3,3>(0,6) = Eigen::Matrix3d::Zero();

// Bottom row: [ -mb*[rbs]× ,   0   ,   I ]
    Zbc.block<3,3>(3,0) = -mass * skew(rs)*-9.81;
    Zbc.block<3,3>(3,3) = Eigen::Matrix3d::Zero();
    Zbc.block<3,3>(3,6) = Eigen::Matrix3d::Identity();



    for (size_t test = 0; test < tags.size(); ++test) {
        auto& accel_data_vec = accel_sets[test];
        auto& orient_data_vec = orient_sets[test];
        auto& wrench_data_vec = wrench_sets[test];

        // Create initial state vector from the first measurements
        Eigen::VectorXd x0(9);
        x0.segment<3>(0) << accel_data_vec[0].ax, accel_data_vec[0].ay, accel_data_vec[0].az;
        x0.segment<3>(3) << wrench_data_vec[0].fx, wrench_data_vec[0].fy, wrench_data_vec[0].fz;
        x0.segment<3>(6) << wrench_data_vec[0].tx, wrench_data_vec[0].ty, wrench_data_vec[0].tz;

        KF.init(0, x0);

        size_t imu_idx = 0, fts_idx = 0, orient_idx = 0;
        long long t_prev = 0;

        // Update process covariance


        while (imu_idx < accel_data_vec.size() || fts_idx < wrench_data_vec.size() || orient_idx < orient_data_vec.size()) {

            // Determine next timestamp
            long long t_next = LLONG_MAX;

            if (imu_idx < accel_data_vec.size()) t_next = std::min(t_next, accel_data_vec[imu_idx].time);
            if (fts_idx < wrench_data_vec.size()) t_next = std::min(t_next, wrench_data_vec[fts_idx].time);
            if (orient_idx < orient_data_vec.size()) t_next = std::min(t_next, orient_data_vec[orient_idx].time);

            double dt = static_cast<double>(t_next - t_prev)/1e2;

            Q.block<3,3>(0,0) = dt* sigma_k*Eigen::Matrix3d::Identity();
            Q.block<3,3>(3,3) =  dt*sigma_k*mass * Eigen::Matrix3d::Identity();
            Q.block<3,3>(6,6) = dt* sigma_k*rs.norm() * Eigen::Matrix3d::Identity();
            KF.setQ(Q);


            t_prev = t_next;

            // Compute control input
            Eigen::Vector3d uk = Eigen::Vector3d::Zero();

            if (orient_idx > 0 && orient_idx < orient_data_vec.size()) {
                uk = compute_uk(orient_data_vec[orient_idx-1],
                                orient_data_vec[orient_idx],
                                fr, ff, fa);
            }

            KF.predict(uk);

            // Update with IMU
            if (imu_idx < accel_data_vec.size() && accel_data_vec[imu_idx].time == t_next) {
                //Eigen::VectorXd za(3);
                //za << accel_data_vec[imu_idx].ax, accel_data_vec[imu_idx].ay, accel_data_vec[imu_idx].az;

                Eigen::Vector3d imu_a(accel_data_vec[imu_idx].ax,
                                      accel_data_vec[imu_idx].ay,
                                      accel_data_vec[imu_idx].az);

                // Rotate to FTS frame using your fixed rotation matrix
                Eigen::Vector3d a_s = R_fa * imu_a;

                Eigen::VectorXd za(3);
                za << a_s(0), a_s(1), a_s(2);
                KF.update(za, Ha, Ra);
                KF.update(za, Ha, Ra);
                imu_idx++;

            }

            // Update with FTS
            if (fts_idx < wrench_data_vec.size() && wrench_data_vec[fts_idx].time == t_next) {
                Eigen::VectorXd zf(6);
                zf << wrench_data_vec[fts_idx].fx, wrench_data_vec[fts_idx].fy, wrench_data_vec[fts_idx].fz,
                        wrench_data_vec[fts_idx].tx, wrench_data_vec[fts_idx].ty, wrench_data_vec[fts_idx].tz;
                KF.update(zf, Hf, Rf);
                fts_idx++;
            }

            // Advance orientation index
            if (orient_idx < orient_data_vec.size() && orient_data_vec[orient_idx].time == t_next) {
                orient_idx++;
            }

            // Store KF estimate with timestamp
            Eigen::Matrix<double,6,1> zbc = Zbc * KF.getXHat();

            Eigen::Vector3d compensated_force =  zbc.segment<3>(0);
            Eigen::Vector3d compensated_torque = zbc.segment<3>(3);

            kf_estimates[test].push_back({t_next, KF.getXHat(),compensated_force,compensated_torque});

        }
    }

    // ===== Save KF filtered data =====
    for (size_t i = 0; i < tags.size(); ++i) {
        std::string tag_name = "Filtered_data/" + std::filesystem::path(tags[i]).filename().string();
        save_kf_csv(tag_name + "_kf.csv", kf_estimates[i]);
    }

    std::cout << "All data saved to Filtered_data/\n";
    return 0;

}
