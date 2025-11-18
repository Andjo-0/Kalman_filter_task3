#ifndef KALMAN_FILTER_KALMAN_HPP
#define KALMAN_FILTER_KALMAN_HPP

#endif //KALMAN_FILTER_KALMAN_HPP

#include <Eigen/Dense>


class kalman_filter{


public:

    kalman_filter(const double& dt,
                  Eigen::MatrixXd A,
                  Eigen::MatrixXd B,
                  Eigen::MatrixXd Q,
                  Eigen::MatrixXd R,
                  Eigen::MatrixXd H,
                  Eigen::MatrixXd P);

    kalman_filter();

    void predict(const Eigen::VectorXd& u);

    void update(const Eigen::VectorXd& z,Eigen::MatrixXd H,Eigen::MatrixXd R);

    void init(int t0, const Eigen::VectorXd& x0);

    Eigen::VectorXd getPrediction(const Eigen::VectorXd& u,const Eigen::VectorXd& z);

    void setH(const Eigen::MatrixXd &h);
    void setA(const Eigen::MatrixXd &a);
    void setB(const Eigen::MatrixXd &b);
    void setQ(const Eigen::MatrixXd &q);
    void setR(const Eigen::MatrixXd &r);

    const Eigen::VectorXd &getXHat() const;


private:

    //Sytem matrix A and Input matrix B
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;

    //Covarianve matrices
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;

    //Prediction matrix??
    Eigen::MatrixXd H_;
    Eigen::MatrixXd P_;


    //Kalman gain matrix

    Eigen::MatrixXd K_;
    int n_;
    Eigen::MatrixXd I_;

    //State vector x, input vector u and measurement vector Z.

    Eigen::VectorXd x_hat;

    Eigen::VectorXd x_hat_new;

    double dt_;
    double t_;

    std::vector<Eigen::MatrixXd> x_estimates;
    std::vector<Eigen::MatrixXd> z_values;

};