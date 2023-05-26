#ifndef ECSPACE_KALMAN_H
#define ECSPACE_KALMAN_H

#include <Eigen/Dense>

/// This code defines three classes: `KalmanFilter`, `ExtendedKalmanFilter`,
/// and `UnscentedKalmanFilter` that are templated on the types of the state
/// vector, control input vector, measurement vector, and matrix. This
/// allows the use of different implementations of matrices and vectors,
/// such as Eigen or Armadillo.
template <typename Matrix, typename Vector>
class KalmanFilter {
public:
    // Constructor for setting the initial state, state transition model, observation model, process noise covariance and observation noise covariance
    KalmanFilter(const Vector& x0, const Matrix& F, const Matrix& H, const Matrix& Q, const Matrix& R);

    // Predict the state and the state covariance
    void predict();

    // Update the state and the state covariance with a new observation
    void update(const Vector& z);

    // Get the current state estimate
    Vector state() const;

    // Get the current state covariance estimate
    Matrix state_covariance() const;

protected:
    Vector x_; // Current state estimate
    Matrix P_; // Current state covariance estimate
    Matrix F_; // State transition model
    Matrix H_; // Observation model
    Matrix Q_; // Process noise covariance
    Matrix R_; // Observation noise covariance
};

template <typename Matrix, typename Vector>
class ExtendedKalmanFilter : public KalmanFilter<Matrix, Vector> {
public:
    // Constructor for setting the initial state, state transition model, observation model, process noise covariance and observation noise covariance
    ExtendedKalmanFilter(const Vector& x0, const Matrix& F, const Matrix& H, const Matrix& Q, const Matrix& R);

    // Function to predict the state and the state covariance
    void predict(const Vector& u);

    // Function to update the state and the state covariance with a new observation
    void update(const Vector& z);

protected:
    // Function to calculate the Jacobian matrices for the state transition and observation models
    virtual Matrix state_jacobian(const Vector& x) = 0;
    virtual Matrix observation_jacobian(const Vector& x) = 0;
};

template <typename Matrix, typename Vector>
ExtendedKalmanFilter<Matrix, Vector>::ExtendedKalmanFilter(const Vector& x0, const Matrix& F, const Matrix& H, const Matrix& Q, const Matrix& R)
    : KalmanFilter<Matrix, Vector>(x0, F, H, Q, R) {
    // Empty constructor
}

template <typename Matrix, typename Vector>
void ExtendedKalmanFilter<Matrix, Vector>::predict(const Vector& u) {
    // Predict the state using the non-linear state transition model
    x_ = state_transition_model(x_, u);
    // Calculate the Jacobian matrix for the state transition model
    Matrix Fx = state_jacobian(x_);
    // Predict the state covariance
    P_ = Fx * P_ * Fx.transpose() + Q_;
}

template <typename Matrix, typename Vector>
void ExtendedKalmanFilter<Matrix, Vector>::update(const Vector& z) {
    // Calculate the Jacobian matrix for the observation model
    Matrix Hx = observation_jacobian(x_);
    // Calculate the Kalman gain
    Matrix K = P_ * Hx.transpose() * (Hx * P_ * Hx.transpose() + R_).inverse();
    // Update the state estimate
    x_ = x_ + K * (z - observationModel(x_));
    // Update the state covariance estimate
    P_ = (Matrix::Identity(x_.size(), x_.size()) - K * Hx) * P_;
}

class EigenExtendedKalmanFilter : public ExtendedKalmanFilter<Eigen::MatrixXd, Eigen::VectorXd> {
public:
    EigenExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& F, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

protected:
    Eigen::MatrixXd state_jacobian(const Eigen::VectorXd& x);
    Eigen::MatrixXd observation_jacobian(const Eigen::VectorXd& x);
    Eigen::VectorXd state_transition_model(const Eigen::VectorXd& x, const Eigen::VectorXd& u);
    Eigen::VectorXd observation_model(const Eigen::VectorXd& x);
};

EigenExtendedKalmanFilter::EigenExtendedKalmanFilter(const Eigen::VectorXd& x0, const Eigen::MatrixXd& F, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
    : ExtendedKalmanFilter<Eigen::MatrixXd, Eigen::VectorXd>(x0, F, H, Q, R) {
    // Empty constructor
}

Eigen::MatrixXd EigenExtendedKalmanFilter::state_jacobian(const Eigen::VectorXd& x) {
    // Calculate the Jacobian matrix for the state transition model
    // Example implementation: return F_;
}

Eigen::MatrixXd EigenExtendedKalmanFilter::observation_jacobian(const Eigen::VectorXd& x) {
    // Calculate the Jacobian matrix for the observation model
    // Example implementation: return H_;
}

Eigen::VectorXd EigenExtendedKalmanFilter::state_transition_model(const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    // Implement the state transition model
    // Example implementation: return F_ * x + u;
}

Eigen::VectorXd EigenExtendedKalmanFilter::observation_model(const Eigen::VectorXd& x) {
    // Implement the observation model
    // Example implementation: return H_ * x;
}
