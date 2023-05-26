#ifndef LINALG_MATRIX_TRAITS_H
#define LINALG_MATRIX_TRAITS_H


template<typename T>
struct matrix_traits {
    using Scalar = typename T::Scalar;
    static constexpr bool is_static = false;
    static constexpr std::size_t rows_at_compile_time = -1;
    static constexpr std::size_t cols_at_compile_time = -1;

    //Return the maximum scalar value in the matrix
    static matrix_traits<T>::Scalar max(const T &mat);

    //Return the transpose of the matrix
    static T transpose(const T &mat);

    //Return the inverse of the matrix
    static T inverse(const T &mat);

    //Return the determinant of the matrix
    static Scalar determinant(const T &mat);

    //Return the trace of the matrix
    static Scalar trace(const T &mat);

    // print
    static void print(const T &mat);


};

#include "matrix_traits_custom_matrix.h"
#include "matrix_traits_eigen_dynamic.h"




//static_assert(T::rows == T::cols, "Matrix must be square.");

// write a linalg struct that uses the matrix_traits and provides
// operations which infer the type of the matrix



// partial specialize for vector_traits, which is a column matrix
//template<typename T>
//struct vector_traits : public matrix_traits<T> {
//    static constexpr std::size_t rows_at_compile_time = -1;
//    static constexpr std::size_t cols_at_compile_time = 1;
//
//    // vector addition
//    friend T operator+(const T &lhs, const T &rhs) {
//        T result = lhs;
//        result += rhs;
//        return result;
//    }
//};

//template<typename Scalar, int Rows, int Cols>
//struct matrix_traits<CustomMatrix<Scalar, Rows, Cols>> {
//    using ScalarType = Scalar;
//    static constexpr int RowsAtCompileTime = Rows;
//    static constexpr int ColsAtCompileTime = Cols;
//
//    using MatrixType = StaticMatrix<Scalar, Rows, Cols>;
//
//    void print(const MatrixType &matrix) {
//        matrix.print();
//    }
//
//    friend MatrixType operator+(const MatrixType &lhs, const MatrixType &rhs) {
//        MatrixType result = lhs;
//        result += rhs;
//        return result;
//    }
//};


//
//template<typename Scalar, int Rows, int Cols>
//struct MatrixTrait<Eigen::Matrix<Scalar, Rows, Cols>,
//                   std::enable_if_t<(Rows > 0 && Cols > 0)>> {
//    using ScalarType = Scalar;
//    static constexpr int RowsAtCompileTime = Rows;
//    static constexpr int ColsAtCompileTime = Cols;
//
//    void print() {
//        std::cout << "Matrix is supported for printing." << std::endl;
//    }
//};

//#include "matrix_traits_eigen_dynamic.h"

#endif //LINALG_MATRIX_TRAITS_H