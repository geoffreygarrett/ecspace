#ifndef MATRIX_TRAITS_EIGEN_FIXED_H
#define MATRIX_TRAITS_EIGEN_FIXED_H

#include <Eigen/Core>

template <typename ScalarType, int Rows, int Cols>
struct MatrixTraits<Eigen::Matrix<ScalarType, Rows, Cols>> {
    using MatrixType = Eigen::Matrix<ScalarType, Rows, Cols>;
    using ScalarType = typename MatrixType::Scalar;
    static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
    static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    // Specializations for fixed-sized Eigen matrices
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
