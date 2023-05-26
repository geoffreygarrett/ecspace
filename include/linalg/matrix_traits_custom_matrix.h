//
// Created by ggarrett on 1/19/23.
//

#ifndef LINALG_TRAITS_MATRIX_TRAITS_CUSTOM_MATRIX_H
#define LINALG_TRAITS_MATRIX_TRAITS_CUSTOM_MATRIX_H

#include "matrix_traits.h"

// implement for custom matrix
template<typename ScalarType, int Rows, int Cols>
struct matrix_traits<CustomMatrix<ScalarType, Rows, Cols>> {
    using MatrixType = CustomMatrix<ScalarType, Rows, Cols>;
    using Scalar = ScalarType;
    static constexpr bool is_static = true;
    static constexpr std::size_t rows_at_compile_time = Rows;
    static constexpr std::size_t cols_at_compile_time = Cols;

    static Scalar max(const MatrixType &mat) {
        // find max by iterating
        Scalar max = mat(0, 0);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                if (mat(i, j) > max) {
                    max = mat(i, j);
                }
            }
        }
        return max;
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
    static void print(const MatrixType &mat) {
        mat.print();
    }
};
#endif//LINALG_TRAITS_MATRIX_TRAITS_CUSTOM_MATRIX_H
