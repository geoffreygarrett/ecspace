#ifndef MATRIX_TRAITS_H
#define MATRIX_TRAITS_H

template <typename Matrix>
struct MatrixTraits
{
    using Scalar = typename Matrix::Scalar;
    static constexpr int Rows = -1;
    static constexpr int Cols = -1;
    //Return the maximum scalar value in the matrix
    static Scalar max(const Matrix& mat);
    //Return the transpose of the matrix
    static Matrix transpose(const Matrix& mat);
    //Return the inverse of the matrix
    static Matrix inverse(const Matrix& mat);
    //Return the determinant of the matrix
    static Scalar determinant(const Matrix& mat);
    //Return the trace of the matrix
    static Scalar trace(const Matrix& mat);
    //Return the rank of the matrix
    static int rank(const Matrix& mat);
    //Solve the least squares problem for mat * x = vec using SVD decomposition
    static Matrix solveLeastSquares(const Matrix& mat, const Matrix& vec);
    //Matrix addition
    Matrix operator+(const Matrix& mat) const;
    //Matrix subtraction
    Matrix operator-(const Matrix& mat) const;
    //Matrix multiplication
    Matrix operator*(const Matrix& mat) const;
    //Scalar multiplication
    Matrix operator*(const Scalar& scalar) const;
    //Scalar division
    Matrix operator/(const Scalar& scalar) const;
};

#include "matrix_traits_eigen_dynamic.h"
#include "matrix_traits_eigen_fixed.h"
#include "matrix_traits_cuda.h"

#endif
