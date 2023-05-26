#ifndef MATRIX_TRAITS_EIGEN_DYNAMIC_H
#define MATRIX_TRAITS_EIGEN_DYNAMIC_H

#include <Eigen/Core>

template <typename ScalarType>
struct MatrixTraits<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>> {
    using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    using ScalarType = typename MatrixType::Scalar;
    static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
    static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    // Specializations for dynamic-sized Eigen matrices
    static MatrixType add(const MatrixType &a, const MatrixType &b) {
        return a + b;
    }

    static MatrixType subtract(const MatrixType &a, const MatrixType &b) {
        return a - b;
    }

    static MatrixType multiply(const MatrixType &a, const MatrixType &b) {
        return a * b;
    }

    static MatrixType transpose(const MatrixType &m) {
        return m.transpose();
    }

    static ScalarType determinant(const MatrixType &m) {
        return m.determinant();
    }

    static MatrixType inverse(const MatrixType &m) {
        return m.inverse();
    }

    static MatrixType solve(const MatrixType &A, const MatrixType &b) {
        return A.colPivHouseholderQr().solve(b);
    }
};

#endif
