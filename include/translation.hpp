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

#include <concepts>
#include <cstddef>

#include "vector.hpp"
#include "rotation.hpp"
#include "transform_conventions.hpp"

namespace zdm::la
{

template <std::floating_point T, std::size_t N, Action action_param = Action::passive>
class Translation
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using vector_type = Vector<T, N>;

    static constexpr Action action = action_param;

    constexpr Translation() = default;

    explicit constexpr Translation(const vector_type& vector):
        m_translation{vector} {}

    [[nodiscard]] static constexpr Translation
    identity() noexcept
    {
        return Translation{vector_type{}};
    }

    [[nodiscard]] static constexpr vector_type
    from(const vector_type& vector) noexcept
    {
        return Translation{vector};
    }

    [[nodiscard]] explicit constexpr
    operator vector_type() const noexcept
    { 
        return (action == Action::active) ? m_translation : -m_translation;
    }

    [[nodiscard]] constexpr vector_type
    operator()(const vector_type& vector) const noexcept
    {
        if constexpr (action == Action::active)
            return vector + m_translation;
        else
            return vector - m_translation;
    }

    [[nodiscard]] friend constexpr Translation
    operator+(const Translation& a) noexcept
    {
        return a;
    }

    [[nodiscard]] friend constexpr Translation
    operator+(const Translation& a, const Translation& b) noexcept
    {
        return Translation{a.m_translation + b.m_translation};
    }

    [[nodiscard]] friend constexpr Translation
    operator-(const Translation& a) noexcept
    {
        return Translation{-a.m_translation};
    }

    [[nodiscard]] friend constexpr Translation
    operator-(const Translation& a, const Translation& b) noexcept
    {
        return Translation{a.m_translation - b.m_translation};
    }

    [[nodiscard]] friend constexpr Translation
    operator*(const vector_type::value_type& a, const Translation& b) noexcept
    {
        return Translation{a*b.m_translation};
    }

    [[nodiscard]] friend constexpr Translation
    operator*(const Translation& a, const vector_type::value_type& b) noexcept
    {
        return Translation{a.m_translation*b};
    }

    template <MatrixLayout layout>
    [[nodiscard]] friend constexpr Translation
    operator*(const RotationMatrix<T, N, action, layout>& a, const Translation& b) noexcept
    {
        return Translation{a*b.m_translation};
    }

    [[nodiscard]] constexpr Translation
    inverse() const noexcept
    {
        return Translation{-m_translation};
    }

private:
    vector_type m_translation;
};

/**
    @brief Compose two translations.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.

    @param a
    @param b

    @return Composite translation.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action
>
[[nodiscard]] constexpr Translation<T, N, action>
compose(const Translation<T, N, action>& a, const Translation<T, N, action>& b)
{
    return a + b;
}

} // namespace zdm::la
