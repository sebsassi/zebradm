/*
Copyright (c) 2024 Sebastian Sassi

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
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <span>
#include <stdexcept>

#include "numerical/linalg.hpp"

namespace coordinates
{

template <typename TransformType>
void transform(
    std::span<Vector<double, 3>>& vectors, const std::span<double>& time_interval, TransformType transformation) noexcept
{
    if (vectors.size() != time_interval.size())
        throw std::invalid_argument(
                "vectors and time_interval must have the same size");
    
    for (std::size_t i = 0; i < time_interval.size(); ++i)
    {
        Vector<double, 3>& v = vectors[i];
        const Vector<double, 4> v_aug = {v[0], v[1], v[2], 1.0};
        const Matrix<double, 4, 4> aff = transformation.transform(time_interval[i]);
        const Vector<double, 4> v_trans = matmul(aff, v_aug);
        v = {v_trans[0], v_trans[1], v_trans[2]};
    }
}

template <typename TransformType>
void rotate(
    std::span<Vector<double, 3>>& vectors, const std::span<double>& time_interval, TransformType transformation) noexcept
{
    if (vectors.size() != time_interval.size())
        throw std::invalid_argument(
                "vectors and time_interval must have the same size");
    
    for (std::size_t i = 0; i < time_interval.size(); ++i)
    {
        Vector<double, 3>& v = vectors[i];
        const Vector<double, 3> v_copy = v;
        const Matrix<double, 3, 3> rot = transformation.rotation(time_interval[i]);
        v = matmul(rot, v_copy);
    }
}

template <typename TransformType>
void rotate(
    std::span<Matrix<double, 3, 3>>& matrices, const std::span<double>& time_interval, TransformType transformation) noexcept
{
    if (matrices.size() != time_interval.size())
        throw std::invalid_argument(
                "matrices and time_interval must have the same size");
    
    for (std::size_t i = 0; i < time_interval.size(); ++i)
    {
        Vector<double, 3>& mat = matrices[i];
        const Matrix<double, 3, 3> rot = transformation.rotation(time_interval[i]);
        mat = matmul(aff, matmul(mat, transpose(aff)));
    }
}

template <typename TransformType>
class InverseRigidTransform
{
public:
    constexpr InverseRigidTransform(TransformType p_transform) : m_transform(p_transform) {}

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> transform(double time) const noexcept
    {
        return invert_rigid(m_transform.transform(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> inverse_transform(double time) const noexcept
    {
        return m_transform.transform(time);
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation(double time) const noexcept
    {
        return transpose(m_transform.rotation(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> inverse_rotation(double time) const noexcept
    {
        return m_transform.rotation(time);
    }

private:
    TransformType m_transform;
};

template <typename TransformType>
using InverseGalileanTransform = InverseRigidTransform<TransformType>;

template <typename... TransformTypes>
class CompositeRigidTransform
{
public:
    constexpr CompositeRigidTransform(const TransformTypes&... p_transforms)
    : m_transforms(p_transforms) {}

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> transform(double time) const noexcept
    {
        return compose_transforms(m_transforms, identity<double, 4>(), time);
    }

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> inverse_transform(double time) const noexcept
    {
        return invert_rigid(transform(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation(double time) const noexcept
    {
        return compose_rotations(m_transforms, identity<double, 3>(), time);
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> inverse_rotation(double time) const noexcept
    {
        return transpose(rotation(time));
    }

private:
    std::tuple<TransformTypes...> m_transforms;
};

template <typename... TransformTypes>
using CompositeGalileanTransform = CompositeRigidTransform<TransformTypes...>;

template<std::size_t I = 0, typename... Tp>
[[nodiscard]] constexpr Matrix<double, 4, 4> compose_transforms(
    std::tuple<Tp...>& t, Matrix<double, 4, 4> x, double time) noexcept
{
    if constexpr (I < sizeof...(Tp))
        return compose_transforms<I + 1>(
                t, matmul(std::get<I>(t).transform(time), x), time);
    else
        return x;
}

template<std::size_t I = 0, typename... Tp>
[[nodiscard]] constexpr Matrix<double, 3, 3> compose_rotations(
    std::tuple<Tp...>& t, Matrix<double, 3, 3> x, double time) noexcept
{
    if constexpr (I < sizeof...(Tp))
        return compose_rotations<I + 1>(t, matmul(std::get<I>(t).rotation(time), x), time);
    else
        return x;
}

template<std::size_t I = 0, typename FieldType, typename... Tp>
[[nodiscard]] FieldType compose_tuple(std::tuple<Tp...>& t, FieldType x) noexcept
{
    if constexpr (I < sizeof...(Tp))
        return compose_tuple<I + 1>(t, std::get<I>(t)(x));
    else
        return x;
}

}