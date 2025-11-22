// read_calibrated_quadratic.cpp
//
// Read serial data from COM port (space-separated tokens per line)
// Parse calibration CSV with columns: C, L_s0..L_s7, Q_s0s0, Q_s0s1, ... Q_s7s7
// Compute W = C + L*S + Q*quad_terms and print at ~200 Hz.
//
// Dependencies:
//   - Boost (Boost.Asio) for serial port
//   - Eigen for matrix math
//
// Build (example for g++ on Linux):
//   g++ -std=c++17 read_calibrated_quadratic.cpp -I/path/to/eigen -lboost_system -lpthread -o read_calibrated_quadratic
//
// On Windows (MSVC) link with Boost libraries accordingly.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>

#include <boost/asio.hpp>
#include <boost/asio/serial_port.hpp>

#include <Eigen/Dense>

using boost::asio::serial_port_base;
namespace asio = boost::asio;

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool load_calibration_csv(const std::string &path,
                          Eigen::VectorXd &C,        // (6)
                          Eigen::MatrixXd &L,        // (6 x 8)
                          Eigen::MatrixXd &Q) {      // (6 x 36 - triangular ordering)
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open calibration CSV: " << path << std::endl;
        return false;
    }

    std::string header;
    if (!std::getline(ifs, header)) {
        std::cerr << "Empty calibration file." << std::endl;
        return false;
    }

    auto cols = split(header, ',');
    for (auto &col : cols) col = trim(col);

    // Expected: columns include "C", "L_s0".."L_s7", "Q_s0s0" ... triangular Q entries
    // We'll read the first (and only) data row and map by column name.
    std::string line;
    if (!std::getline(ifs, line)) {
        std::cerr << "Calibration file has no data rows." << std::endl;
        return false;
    }
    auto values = split(line, ',');
    for (auto &v : values) v = trim(v);

    if (cols.size() != values.size()) {
        std::cerr << "Header/value column mismatch in CSV." << std::endl;
        return false;
    }

    // Create a map col->value (string)
    std::unordered_map<std::string, std::string> mapcv;
    for (size_t i = 0; i < cols.size(); ++i) {
        mapcv[cols[i]] = values[i];
    }

    // Parse C (one per wrench component). Your Python saved df with rows for each wrench (Fx..Mz).
    // The CSV probably has 6 rows in Python; but the Python code constructed a DataFrame where each row is a wrench
    // and then wrote params_...csv. That file likely has 6 rows. To handle both cases:
    // - If CSV has 6 rows, we need to parse all rows into vectors. But above we read only first data row.
    // So to be robust, instead re-open and parse as table with columns. We'll do that below.

    // Reset file and parse as table of rows (each row corresponds to a wrench: Fx,..Mz)
    ifs.clear();
    ifs.seekg(0);
    // Read header again
    std::getline(ifs, header);
    std::vector<std::vector<std::string>> table;
    while (std::getline(ifs, line)) {
        if (trim(line).empty()) continue;
        auto row = split(line, ',');
        for (auto &r : row) r = trim(r);
        table.push_back(row);
    }

    if (table.size() < 6) {
        std::cerr << "Calibration CSV should contain 6 rows (Fx,Fy,Fz,Mx,My,Mz). Found: " << table.size() << std::endl;
        return false;
    }

    // Find column indices
    auto find_col = [&](const std::string &name)->int{
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] == name) return (int)i;
        }
        return -1;
    };

    // Initialize outputs
    C = Eigen::VectorXd::Zero(6);
    L = Eigen::MatrixXd::Zero(6, 8);
    Q = Eigen::MatrixXd::Zero(6, 36);

    // For each wrench row (0..5)
    for (int r = 0; r < 6; ++r) {
        const auto &row = table[r];
        // C
        int ci = find_col("C");
        if (ci >= 0 && ci < (int)row.size()) {
            C(r) = std::stod(row[ci]);
        } else {
            std::cerr << "Column C not found.\n";
            return false;
        }
        // L_s0..L_s7
        for (int s = 0; s < 8; ++s) {
            std::string name = "L_s" + std::to_string(s);
            int idx = find_col(name);
            if (idx >= 0 && idx < (int)row.size()) {
                L(r, s) = std::stod(row[idx]);
            } else {
                std::cerr << "Column " << name << " not found.\n";
                return false;
            }
        }
        // Q triangular entries in the same ordering: for i in 0..7 for j in i..7 -> total 36
        int qcol = 0;
        for (int i = 0; i < 8; ++i) {
            for (int j = i; j < 8; ++j) {
                std::string name = "Q_s" + std::to_string(i) + "s" + std::to_string(j);
                int idx = find_col(name);
                if (idx >= 0 && idx < (int)row.size()) {
                    Q(r, qcol) = std::stod(row[idx]);
                } else {
                    std::cerr << "Column " << name << " not found.\n";
                    return false;
                }
                ++qcol;
            }
        }
    }

    return true;
}

// Build the quad_terms vector with ordering identical to the Python code:
// quad_terms = [S[i] * S[j] for i in range(len(S)) for j in range(i, len(S))]
static Eigen::VectorXd build_quad_terms(const Eigen::VectorXd &S) {
    int n = (int)S.size();
    int nterms = n * (n + 1) / 2;
    Eigen::VectorXd q(nterms);
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            q(idx++) = S(i) * S(j);
        }
    }
    return q;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting..." << std::endl;

        // Config (change if needed)
        std::string port = "COM4";          // On Linux use "/dev/ttyUSB0" or similar
        unsigned int baudrate = 115200;
        const int n_sensors = 8;
        const int overload_lower = 0;
        const int overload_upper = 1023;

        // Calibration CSV path (change as required)
        std::string calib_path = "Params/params_lasso_quadratic.csv";

        // Load calibration
        Eigen::VectorXd C;
        Eigen::MatrixXd L, Q;
        if (!load_calibration_csv(calib_path, C, L, Q)) {
            std::cerr << "Failed to load calibration file." << std::endl;
            return 1;
        }
        std::cout << "Loaded calibration (C size " << C.size() << ", L " << L.rows() << "x" << L.cols()
                  << ", Q " << Q.rows() << "x" << Q.cols() << ")" << std::endl;

        // Setup serial
        asio::io_context io;
        asio::serial_port serial(io);
        boost::system::error_code ec;

        serial.open(port, ec);
        if (ec) {
            std::cerr << "Failed to open serial port " << port << ": " << ec.message() << std::endl;
            return 1;
        }

        serial.set_option(serial_port_base::baud_rate(baudrate));
        serial.set_option(serial_port_base::character_size(8));
        serial.set_option(serial_port_base::parity(serial_port_base::parity::none));
        serial.set_option(serial_port_base::stop_bits(serial_port_base::stop_bits::one));
        serial.set_option(serial_port_base::flow_control(serial_port_base::flow_control::none));

        asio::streambuf readbuf;

        std::cout << "Got the serial port." << std::endl;

        while (true) {
            // Read a line
            boost::asio::read_until(serial, readbuf, "\n", ec);
            if (ec) {
                std::cerr << "Serial read error: " << ec.message() << std::endl;
                break;
            }

            std::istream is(&readbuf);
            std::string rawline;
            std::getline(is, rawline);
            rawline = trim(rawline);
            if (rawline.empty()) {
                // avoid busy spinning
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Expected tokenization like Python: tokens separated by whitespace
            std::istringstream tokens(rawline);
            std::vector<std::string> toks;
            std::string tk;
            while (tokens >> tk) toks.push_back(tk);

            // Expect at least 11 tokens: str_D seq_number error_mask s0..s7
            if (toks.size() < 11) {
                std::cerr << "Error parsing data (not enough tokens): '" << rawline << "'" << std::endl;
                continue;
            }

            // Convert tokens: first token is string, next ints
            // In Python you did: (str_D, seq_number, error_mask, s0..s7) = [t(s) for t,s in zip(...)]
            // We'll just parse ints for the numeric tokens
            int seq_number = 0;
            try {
                seq_number = std::stoi(toks[1]);
            } catch (...) { seq_number = 0; }

            std::vector<int> svals(n_sensors);
            bool parse_ok = true;
            try {
                for (int i = 0; i < n_sensors; ++i) {
                    svals[i] = std::stoi(toks[3 + i]); // tokens: 0->str,1->seq,2->errmask, 3..10 -> s0..s7
                }
            } catch (...) {
                std::cerr << "Error parsing sensor ints from: '" << rawline << "'" << std::endl;
                parse_ok = false;
            }
            if (!parse_ok) continue;
            svals[5]=svals[4];

            // Overload check
            bool overloaded = false;
            for (int i = 0; i < n_sensors; ++i) {
                if (svals[i] < overload_lower || svals[i] > overload_upper) {
                    std::cout << "Force overload channel " << i << std::endl;
                    overloaded = true;
                    // do not break - print message and continue to next sample (like Python `continue` inside loop)
                    break;
                }
            }
            if (overloaded) continue;

            // Form Eigen vector S
            Eigen::VectorXd S(n_sensors);
            for (int i = 0; i < n_sensors; ++i) S(i) = static_cast<double>(svals[i]);

            // Compute quad terms and W
            Eigen::VectorXd qterms = build_quad_terms(S); // length 36
            Eigen::VectorXd W = Eigen::VectorXd::Zero(6);
            // linear part
            W = C + L * S;
            // quadratic
            W += Q * qterms;

            // Print with formatting similar to Python
            std::cout << "\n";
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Fx: " << W(0) << ", Fy: " << W(1) << ", Fz: " << W(2)
                      << ", Mx: " << W(3) << ", My: " << W(4) << ", Mz: " << W(5) << ",\n";
            std::cout << "s0: " << svals[0] << ", s1: " << svals[1] << ", s2: " << svals[2] << ", s3: " << svals[3]
                      << ", s4: " << svals[4] << ", s5: " << svals[5] << ", s6: " << svals[6] << ", s7: " << svals[7] << "\n";
            std::cout << "\n";

            // Sleep equivalent to 1/200 s
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        serial.close();
        std::cout << "Finished." << std::endl;
    } catch (std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}


















