#ifndef LINALG_MATRIX_TRAITS_EIGEN_DYNAMIC_H

#include <Eigen/Dense>
#include <iostream>
#include "matrix_traits.h"


template<typename Scalar, int Rows, int Cols>
struct matrix_traits<Eigen::Matrix<Scalar, Rows, Cols>> {
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    static constexpr bool is_static = false;
    static constexpr std::size_t rows_at_compile_time = MatrixType::RowsAtCompileTime;
    static constexpr std::size_t cols_at_compile_time = MatrixType::ColsAtCompileTime;

    static Scalar max(const MatrixType &mat) {
        return mat.maxCoeff();
    }

    static MatrixType transpose(const MatrixType &mat) {
        return mat.transpose();
    }

    static MatrixType inverse(const MatrixType &mat) {
        return mat.inverse();
    }

    static Scalar determinant(const MatrixType &mat) {
        return mat.determinant();
    }

    static Scalar trace(const MatrixType &mat) {
        return mat.trace();
    }

    static void print(const MatrixType &matrix) {
        std::cout << matrix << std::endl;
    }
};

template<>
struct matrix_traits<Eigen::MatrixXd> {
    using Scalar = double;
    static constexpr bool is_static = false;
    static constexpr std::size_t rows_at_compile_time = -1;
    static constexpr std::size_t cols_at_compile_time = -1;

    static double max(const Eigen::MatrixXd &mat) {
        return mat.maxCoeff();
    }

    static Eigen::MatrixXd transpose(const Eigen::MatrixXd &mat) {
        return mat.transpose();
    }

    static Eigen::MatrixXd inverse(const Eigen::MatrixXd &mat) {
        return mat.inverse();
    }

    static double determinant(const Eigen::MatrixXd &mat) {
        return mat.determinant();
    }

    static double trace(const Eigen::MatrixXd &mat) {
        return mat.trace();
    }

    static void print(const Eigen::MatrixXd &mat) {
        std::cout << mat << std::endl;
    }
};

#endif //LINALG_MATRIX_TRAITS_EIGEN_DYNAMIC_H
