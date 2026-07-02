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
    conventional_arithmetic T, std::size_t N, std::size_t M,
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

    template <std::size_t I>
        requires (I < 2)
    static constexpr size_type extent = shape[I];

    std::array<value_type, N*M> array;

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
    operator std::array<value_type, N*M>() const noexcept { return array; }

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
    template <std::size_t K>
    [[nodiscard]] constexpr Matrix<value_type, N, K, action, layout>
    operator*(const Matrix<value_type, M, K, action, layout>& other) const noexcept
    {
        Matrix<value_type, N, K, action, layout> res{};
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < K; ++j)
            {
                for (std::size_t k = 0; k < M; ++k)
                    res[i, j] += (*this)[i, k]*other[k, j];
            }
        }

        return res;

    }

    /**
        @brief Matrix-vector multiplication.
    */
    [[nodiscard]] constexpr Vector<value_type, N>
    operator*(const static_vector_like auto& vector) const noexcept
    {
        Vector<T, shape[0]> res{};
        for (std::size_t i = 0; i < shape[0]; ++i)
        {
            for (std::size_t j = 0; j < shape[1]; ++j)
                res[i] += (*this)[i, j]*vector[j];
        }

        return res;
    }

    [[nodiscard]] constexpr transpose_type
    transpose() const noexcept
    {
        transpose_type res{};
        for (std::size_t i = 0; i < shape[0]; ++i)
        {
            for (std::size_t j = 0; j < shape[1]; ++j)
                res[j, i] = (*this)[i, j];
        }
        return res;
    }

    [[nodiscard]] constexpr value_type
    norm() const noexcept
    {
        return Vector<value_type, N*M>{array}.norm();
    }

    [[nodiscard]] constexpr value_type
    magnitude() const noexcept
    {
        return norm();
    }
};

} // namespace zdm::la
