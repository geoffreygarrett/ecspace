

#ifndef LINALG_TRAITS_LINALG_H
#define LINALG_TRAITS_LINALG_H

#include "linalg/matrix_concepts.h"
#include "linalg/matrix_traits.h"

namespace linalg {

    template<Matrix T>
    static typename T::Scalar max(const T &mat) {
        return matrix_traits<T>::max(mat);
    }

    template<typename T>
    static T transpose(const T &mat) {
        return matrix_traits<T>::transpose(mat);
    }

    template<typename T, std::enable_if_t<matrix_traits<T>::is_static, bool> = true>
    static T inverse(const T &mat) {
        static_assert(matrix_traits<T>::rows_at_compile_time == matrix_traits<T>::cols_at_compile_time, "Matrix must be square.");
        return matrix_traits<T>::inverse(mat);
    }

    template<typename T, std::enable_if_t<!matrix_traits<T>::is_static, bool> = true>
    static T inverse(const T &mat) {
        if (mat.rows() != mat.cols()) {
            throw std::runtime_error("Matrix must be square.");
        }
        return matrix_traits<T>::inverse(mat);
    }

    template<typename T>
    static typename matrix_traits<T>::Scalar determinant(const T &mat) {
        return matrix_traits<T>::determinant(mat);
    }

    template<typename T>
    static typename matrix_traits<T>::Scalar trace(const T &mat) {
        return matrix_traits<T>::trace(mat);
    }

    template<typename T>
    static void print(const T &mat) {
        matrix_traits<T>::print(mat);
    }

    // variadic sum
    template<typename T>
    T sum(T t) {
        return t;
    }

    template<typename T, typename... Args>
    T sum(T t, Args... args) {
        return t + sum(args...);
    }

    template<typename T>
    T product(T t) {
        return t;
    }

    // variadic product
    template<typename T, typename... Args>
    T product(T t, Args... args) {
        return t * product(args...);
    }

};// namespace linalg

#endif//LINALG_TRAITS_LINALG_H
