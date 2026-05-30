/*
Copyright (c) 2025 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#pragma once

#include <array>
#include <cassert>

#include "concepts.hpp"
#include "linalg.hpp"
#include "transform_conventions.hpp"
#include "vector.hpp"

namespace zdm::la
{

/**
    @brief Enum for specifying the memory layout of a matrix.

    This enum denotes the linear layout of a matrix in memory. Given a matrix
    \f[
        \begin{pmatrix}
            A_{11} & A_{12} & A_{13}\\
            A_{21} & A_{22} & A_{23}\\
            A_{31} & A_{32} & A_{33}
        \end{pmatrix},
    \f]
    in row-major order it is stored in memory as
    \f[
        A_{11}, A_{12}, A_{13}, A_{21}, A_{22}, A_{23}, A_{31}, A_{32}, A_{33},
    \f]
    and in column-major order as
    \f[
        A_{11}, A_{21}, A_{31}, A_{12}, A_{22}, A_{32}, A_{13}. A_{23}, A_{33}.
    \f]
*/
enum class MatrixLayout
{
    row_major,
    column_major
};

/**
    @brief A general matrix type.

    @tparam T Type of matrix elements.
    @tparam N Number of matrix rows.
    @tparam M Number of matrix columns.
    @tparam action_param Matrix action convention.
    @tparam layout_param Matrix layout convention.
*/
template <
    arithmetic T, std::size_t N, std::size_t M,
    Action action_param = Action::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
struct Matrix
{
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using transpose_type = Matrix<T, M, N, action_param, layout_param>;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N, M};

    std::array<T, N*M> array;

    /**
        @brief Create an identity matrix.

        @return \f$N\times N\f$ identity matrix.

        @note This function is only defined for square matrices.
    */
    [[nodiscard]] static constexpr Matrix
    identity() noexcept requires (N == M)
    {
        Matrix res{};
        for (std::size_t i = 0; i < N; ++i)
            res[i, i] = 1.0;
        return res;
    }

    /**
        @brief Get the underlying array that stores the elements of the matrix.

        @return Array with \f$NM\f$ elements.
    */
    [[nodiscard]] explicit constexpr 
    operator std::array<T, N*M>() const noexcept { return array; }

    [[nodiscard]] constexpr bool operator==(const Matrix& other) const noexcept = default;

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept
    {
        assert(i < N && j < M);
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept
    {
        assert(i < N && j < M);
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }

    /**
        @brief Matrix-matrix multiplication.
    */
    [[nodiscard]] constexpr Matrix
    operator*(const Matrix& other) const noexcept requires (N == M) { return matmul(*this, other); }

    /**
        @brief Matrix-vector multiplication.
    */
    template <static_vector_like V>
    [[nodiscard]] constexpr Vector<T, N>
    operator*(const V& vector) const noexcept { return matmul(*this, vector); }
};

/**
    @brief Multiply a vector by a non-square matrix.

    @tparam T Matrix type.
    @tparam U Vector type.

    @param mat Matrix.
    @param vec Vector.

    @return Product vector.

    This function returns the product \f$\vec{b} = M\vec{a}\f$.

    @note Given a non-square matrix and an arbitrary vector type, there is no
    natural corresponding return vector type. Therefore this returns a `Vector`.
*/
template <static_matrix_like T, static_vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[0] != T::shape[1]) && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr Vector<typename U::value_type, T::shape[0]>
matmul(const T& mat, const U& vec) noexcept
{
    Vector<typename U::value_type, T::shape[0]> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

/**
    @brief Multiply two non-square matrices.

    @tparam T Matrix type.
    @tparam U Matrix type.

    @param a
    @param b

    @return Matrix product.

    This function returns the product \f$M = M_1M_2\f$.

    @note Given two non-square matrix types, there is no natural corresponding
    return matrix type. Therefore this returns a `Matrix`.
*/
template <static_matrix_like T, static_matrix_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == U::shape[0])
[[nodiscard]] constexpr Matrix<typename T::value_type, T::shape[0], U::shape[1], T::action, T::layout>
matmul(const T& a, const U& b) noexcept
{
    Matrix<typename T::value_type, T::shape[0], U::shape[1], T::action, T::layout> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < U::shape[1]; ++j)
        {
            for (std::size_t k = 0; k < T::shape[1]; ++k)
                res[i, j] += a[i, k]*b[k, j];
        }
    }

    return res;
}

} // namespace zdm::la
