//Inspired by https://github.com/hmartiro/kalman-cpp

#include "kalman.hpp"
#include <Eigen/Dense>

kalman_filter::kalman_filter(
        const double& dt,
        Eigen::MatrixXd A,
        Eigen::MatrixXd B,
        Eigen::MatrixXd Q,
        Eigen::MatrixXd R,
        Eigen::MatrixXd H,
        Eigen::MatrixXd P) :
        A_(A), B_(B), Q_(Q), R_(R), H_(H), P_(P),
        dt_(dt),
        n_(A_.rows()),
        I_(n_, n_),
        x_hat(n_),
        x_hat_new(n_)
{

    I_.setIdentity();
    x_hat.setZero();
    x_hat_new.setZero();
}

void kalman_filter::predict(const Eigen::VectorXd& u) {

    x_hat_new = A_*x_hat + B_*u;
    P_ = A_*P_*A_.transpose()+(Q_);

}

void kalman_filter::update(const Eigen::VectorXd& z,Eigen::MatrixXd H,Eigen::MatrixXd R) {
    setH(H);
    setR(R);

    z_values.push_back(z);

    K_= P_*H_.transpose()*(H_*P_*H_.transpose()+R_).inverse();
    x_hat_new+= K_*(z-H_*x_hat_new);
    P_ = (I_-K_*H_)*P_;
    x_hat = x_hat_new;
    x_estimates.push_back(x_hat);

}

void kalman_filter::init(int t0,const Eigen::VectorXd& x0) {
    x_hat = x0;
    t_=t0;

}

Eigen::VectorXd kalman_filter::getPrediction(const Eigen::VectorXd& u, const Eigen::VectorXd& z) {

    predict(u);
    //update(z);

    return x_hat;
}

void kalman_filter::setA(const Eigen::MatrixXd &a) {
    A_ = a;
}

void kalman_filter::setB(const Eigen::MatrixXd &b) {
    B_ = b;
}

void kalman_filter::setQ(const Eigen::MatrixXd &q) {
    Q_ = q;
}

void kalman_filter::setR(const Eigen::MatrixXd &r) {
    R_ = r;
}

void kalman_filter::setH(const Eigen::MatrixXd &h) {
    H_ = h;
}

const Eigen::VectorXd &kalman_filter::getXHat() const {
    return x_hat;
}




