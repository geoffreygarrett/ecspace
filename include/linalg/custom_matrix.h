//
// Created by ggarrett on 1/19/23.
//

#ifndef LINALG_TRAITS_CUSTOM_MATRIX_H
#define LINALG_TRAITS_CUSTOM_MATRIX_H

#include <iostream>


template <typename ScalarType, int Rows, int Cols>
class CustomMatrix {
public:
    using Scalar = ScalarType;
    static constexpr int RowsAtCompileTime = Rows;
    static constexpr int ColsAtCompileTime = Cols;

    CustomMatrix() {}

    template <typename OtherScalar, int OtherRows, int OtherCols>
    CustomMatrix(const CustomMatrix<OtherScalar, OtherRows, OtherCols>& other) {
        static_assert(Rows == OtherRows && Cols == OtherCols,
                      "Incompatible matrix size");
        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Cols; j++) {
                data[i][j] = other(i, j);
            }
        }
    }

    template <typename OtherScalar, int OtherRows, int OtherCols>
    CustomMatrix& operator=(const CustomMatrix<OtherScalar, OtherRows, OtherCols>& other) {
        static_assert(Rows == OtherRows && Cols == OtherCols,
                      "Incompatible matrix size");
        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Cols; j++) {
                data[i][j] = other(i, j);
            }
        }
        return *this;
    }

    const Scalar& operator()(int row, int col) const {
        return data[row][col];
    }

    Scalar& operator()(int row, int col) {
        return data[row][col];
    }

    void print() const {
        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Cols; j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    CustomMatrix<Scalar, Rows, Cols> &operator+=(const CustomMatrix<Scalar, Rows, Cols> &other) {
        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Cols; j++) {
                data[i][j] += other.data[i][j];
            }
        }
        return *this;
    }

    // inverse
    CustomMatrix<Scalar, Rows, Cols> inverse() const {
        CustomMatrix<Scalar, Rows, Cols> result;


        return result;
    }

    // transpose
    CustomMatrix<Scalar, Cols, Rows> transpose() const {
        CustomMatrix<Scalar, Cols, Rows> result;
        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Cols; j++) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    // + operator
    CustomMatrix<Scalar, Rows, Cols> operator+(const CustomMatrix<Scalar, Rows, Cols> &other) const {
        CustomMatrix<Scalar, Rows, Cols> result = *this;
        result += other;
        return result;
    }

    // calculate determinant
    Scalar determinant() const {
        Scalar result = 0;
        return result;
    }

    // trace
    Scalar trace() const {
        Scalar result = 0;
        for (int i = 0; i < Rows; i++) {
            result += data[i][i];
        }
        return result;
    }


private:
    Scalar data[Rows][Cols];
};

#endif//LINALG_TRAITS_CUSTOM_MATRIX_H
