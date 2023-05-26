

#ifndef LINALG_TRAITS_MATRIX_CONCEPTS_H
#define LINALG_TRAITS_MATRIX_CONCEPTS_H

#pragma once

#include "matrix_traits.h"
#include <concepts>
#include <cstddef>
#include <type_traits>

template<typename T>
concept Scalar =
        std::is_arithmetic<T>::value ||
        requires(T a, T b) {
            { a + b } -> std::convertible_to<T>;
            { a - b } -> std::convertible_to<T>;
            { a *b } -> std::convertible_to<T>;
            { a / b } -> std::convertible_to<T>;
            { -a } -> std::convertible_to<T>;
            { a == b } -> std::convertible_to<bool>;
            { a != b } -> std::convertible_to<bool>;
        };

template<typename T>
concept Matrix = requires(T a) {
                     { matrix_traits<T>::rows_at_compile_time } -> std::integral;
                     { matrix_traits<T>::cols_at_compile_time } -> std::integral;
                     { matrix_traits<T>::inverse(a) } -> std::convertible_to<T>;
                     { matrix_traits<T>::rows_at_compile_time != -1 } -> std::convertible_to<bool>;
                     { matrix_traits<T>::cols_at_compile_time != -1 } -> std::convertible_to<bool>;
                     { matrix_traits<T>::max(a) } -> Scalar;
                     { matrix_traits<T>::determinant(a) } -> Scalar;
                     { matrix_traits<T>::trace(a) } -> Scalar;
                     { matrix_traits<T>::print(a) } -> std::same_as<void>;
                     { matrix_traits<T>::transpose(a) } -> std::convertible_to<T>;
                 };


#endif//LINALG_TRAITS_MATRIX_CONCEPTS_H
