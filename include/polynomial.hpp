/*
Copyright (c) 2024-2025 Sebastian Sassi

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
#include <concepts>
#include <ranges>
#include <span>
#include <vector>

#include "concepts.hpp"

namespace zdm
{

template <typename S, typename T>
concept subfield_of = requires (S x, T y)
{
    { x*y } -> std::same_as<T>;
    { x + y } -> std::same_as<T>; 
};

namespace detail
{

template <typename T, typename S, typename U>
concept fma_operable = requires (T x, S y, U z)
{
    x + y*z;
};

template <typename R, typename DomainType, typename ResultType>
    requires fma_operable<std::ranges::range_value_t<R>, ResultType, DomainType>
        && requires { std::tuple_size_v<R>; }
[[nodiscard]] constexpr auto
horner_eval(const R& coeffs, const DomainType& x) noexcept
{
    constexpr auto size = std::tuple_size_v<R>;

    if constexpr (size == 0)
        return ResultType{};

    auto res = ResultType{coeffs[size - 1]};
    for (std::size_t i = size - 1; i > 0; --i)
        res = coeffs[i - 1] + res*x;
    return res;
}

template <typename R, typename DomainType, typename ResultType>
    requires fma_operable<std::ranges::range_value_t<R>, std::ranges::range_value_t<R>, DomainType>
        && requires { std::tuple_size_v<R>; }
[[nodiscard]] constexpr auto
estrin_eval(const R& coeffs, const DomainType& x)
{
    constexpr std::size_t size = std::tuple_size_v<R>;
    if constexpr (size == 0)
        return ResultType{};
    if constexpr (size == 1)
        return ResultType{coeffs[0]};
    if constexpr (size == 2)
        return ResultType{coeffs[0] + coeffs[1]*x};

    constexpr std::size_t order = size - 1;
    constexpr std::size_t estrin_order = order >> 1;

    std::array<ResultType, estrin_order + 1> estrin_coeffs{};
    for (std::size_t i = 0; i < estrin_order; ++i)
        estrin_coeffs[i] = ResultType{coeffs[2*i] + coeffs[2*i + 1]*x};

    if constexpr ((size & 1) == 0)
        estrin_coeffs[estrin_order] = ResultType{coeffs[2*estrin_order] + coeffs[2*estrin_order + 1]*x};
    else
        estrin_coeffs[estrin_order] = ResultType{coeffs[2*estrin_order]};

    return estrin_eval<std::array<ResultType, estrin_order + 1>, DomainType, ResultType>(estrin_coeffs, x*x);
}

template <typename R, typename DomainType, typename ResultType>
    requires fma_operable<std::ranges::range_value_t<R>, ResultType, DomainType>
[[nodiscard]] constexpr auto
horner_eval(const R& coeffs, const DomainType& x) noexcept
{
    if (coeffs.size() == 0)
        return ResultType{};
    else
    {
        ResultType res{};
        for (std::size_t i = coeffs.size(); i > 0; --i)
            res = coeffs[i - 1] + res*x;
        return res;
    }
}

} // namespace detail

template <std::size_t N, typename T>
    requires (N > 0) && requires (T x) { { x*x } -> std::same_as<T>; }
T intpow(T x)
{
    if constexpr (N == 1)
        return x;
    else if constexpr (N & 1)
        return x*intpow<(N >> 1)>(x*x);
    else
        return intpow<(N >> 1)>(x*x);
}

template <typename T, std::size_t order_param>
struct Monomial
{
    using value_type = T;

    static constexpr std::size_t order = order_param;

    value_type coeff;

    constexpr Monomial() = default;
    explicit constexpr Monomial(const value_type& coeff_): coeff(coeff_) {}

    template <subfield_of<value_type> DomainType>
    [[nodiscard]] constexpr value_type
    operator()(DomainType x) const noexcept
    {
        if (order == 0)
            return coeff;
        else
            return coeff*intpow<order>(x);
    }

    [[nodiscard]] constexpr auto
    derivative() const noexcept requires zdm::arithmetic<value_type>
    {
        if constexpr (order < 1)
            return Monomial<value_type, 0>{};
        else if constexpr (order == 1)
            return Monomial<value_type, 0>{coeff};
        else
            return Monomial<value_type, order - 1>{value_type(order)*coeff};
    }
};

template <typename ValueType, std::size_t order_param>
struct Polynomial
{
    using value_type = ValueType;
    static constexpr std::size_t order = order_param;

    std::array<value_type, order + 1> coeffs;

    constexpr Polynomial() = default;
    explicit constexpr Polynomial(const std::array<value_type, order + 1>& coeffs_): coeffs(coeffs_) {}

    template <typename... Types>
        requires (std::convertible_to<Types, value_type> && ...)
    constexpr Polynomial(Types... coeffs_): coeffs{value_type(coeffs_)...} {}

    [[nodiscard]] constexpr bool operator==(const Polynomial& other) const noexcept = default;

    template <subfield_of<value_type> DomainType>
    [[nodiscard]] constexpr value_type
    operator()(DomainType x) const noexcept
    {
        return detail::estrin_eval<const std::array<value_type, order + 1>&, DomainType, value_type>(coeffs, x);
    }

    [[nodiscard]] constexpr auto
    derivative() const noexcept requires zdm::arithmetic<value_type>
    {
        if constexpr (order < 1)
            return Polynomial<ValueType, 0>{};
        else if constexpr (order == 1)
            return Polynomial<ValueType, 0>{std::array<ValueType, 1>{coeffs[1]}};
        else
        {
            Polynomial<ValueType, order - 1> deriv{};
            for (std::size_t i = 1; i <= order; ++i)
                deriv.coeffs[i - 1] = ValueType(i)*coeffs[i];
            return deriv;
        }
    }

    template <std::size_t new_order>
        requires (new_order <= order_param)
    [[nodiscard]] constexpr Polynomial<value_type, new_order>
    truncate() const noexcept
    {
        Polynomial<ValueType, new_order> res{};
        for (std::size_t i = 0; i <= new_order; ++i)
            res.coeffs[i] = coeffs[i];
        return res;
    }
};

template <typename T, typename... Types>
    requires (std::same_as<T, Types> && ...)
Polynomial(T, Types...) -> Polynomial<T, sizeof...(Types)>;

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator-(const Polynomial<T, Order>& p) noexcept
{
    Polynomial<T, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = -p.coeffs[i];
    return res;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr auto
operator+(const Polynomial<T, N>& p, const Polynomial<T, M>& q) noexcept
{
    constexpr std::size_t min_order = std::min(N, M);
    constexpr std::size_t max_order = std::max(N, M);

    Polynomial<T, max_order> res{};
    for (std::size_t i = 0; i <= min_order; ++i)
        res.coeffs[i] = p.coeffs[i] + q.coeffs[i];

    for (std::size_t i = min_order + 1; i <= max_order; ++i)
    {
        if constexpr (N > M)
            res.coeffs[i] = p.coeffs[i];
        else
            res.coeffs[i] = q.coeffs[i];
    }

    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator+(const T a, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, order> res{};
    res.coeffs[0] = a + p.coeffs[0];
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];
    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator+(const Polynomial<T, order>& p, const T a) noexcept
{
    Polynomial<T, order> res{};
    res.coeffs[0] = p.coeffs[0] + a;
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];
    return res;
}

template <typename T, std::size_t mon_order, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, std::max(order, mon_order)>
operator+(const Monomial<T, mon_order> q, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, std::max(order, mon_order)> res{};
    for (std::size_t i = 0; i < p.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];

    res.coeffs[mon_order] += q.coeff;

    return res;
}

template <typename T, std::size_t order, std::size_t mon_order>
[[nodiscard]] constexpr Polynomial<T, std::max(order, mon_order)>
operator+(const Polynomial<T, order>& p, const Monomial<T, mon_order> q) noexcept
{
    Polynomial<T, std::max(order, mon_order)> res{};
    for (std::size_t i = 0; i < p.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];

    res.coeffs[mon_order] += q.coeff;

    return res;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr auto
operator-(const Polynomial<T, N>& p, const Polynomial<T, M>& q) noexcept
{
    constexpr std::size_t min_order = std::min(N, M);
    constexpr std::size_t max_order = std::max(N, M);

    Polynomial<T, max_order> res{};
    for (std::size_t i = 0; i <= min_order; ++i)
        res.coeffs[i] = p.coeffs[i] - q.coeffs[i];
 
    for (std::size_t i = min_order + 1; i <= max_order; ++i)
    {
        if constexpr (N > M)
            res.coeffs[i] = p.coeffs[i];
        else
            res.coeffs[i] = -q.coeffs[i];
    }

    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator-(const T a, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, order> res{};
    res.coeffs[0] = a - p.coeffs[0];
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];
    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator-(const Polynomial<T, order>& p, const T a) noexcept
{
    Polynomial<T, order> res{};
    res.coeffs[0] = p.coeffs[0] - a;
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];
    return res;
}

template <typename T, std::size_t mon_order, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, std::max(order, mon_order)>
operator-(const Monomial<T, mon_order> q, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, std::max(order, mon_order)> res{};
    res.coeffs[mon_order] = q.coeff;
    for (std::size_t i = 0; i < p.coeffs.size(); ++i)
        res.coeffs[i] -= p.coeffs[i];


    return res;
}

template <typename T, std::size_t order, std::size_t mon_order>
[[nodiscard]] constexpr Polynomial<T, std::max(order, mon_order)>
operator-(const Polynomial<T, order>& p, const Monomial<T, mon_order> q) noexcept
{
    Polynomial<T, std::max(order, mon_order)> res{};
    for (std::size_t i = 0; i < p.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i];

    res.coeffs[mon_order] -= q.coeff;

    return res;
}

template <typename T, std::size_t N, std::size_t M>
[[nodiscard]] constexpr Polynomial<T, N + M>
operator*(const Polynomial<T, N>& p, const Polynomial<T, M>& q) noexcept
{
    Polynomial<T, N + M> res{};
    for (std::size_t i = 0; i < std::min(N, M); ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
            res.coeffs[i] += p.coeffs[j]*q.coeffs[i - j];
    }

    for (std::size_t i = std::min(N, M); i < std::max(N, M); ++i)
    {
        if constexpr (N < M)
        {
            for (std::size_t j = 0; j <= N; ++j)
                res.coeffs[i] += p.coeffs[j]*q.coeffs[i - j];
        }
        else
        {
            for (std::size_t j = i - M; j <= i; ++j)
                res.coeffs[i] += p.coeffs[j]*q.coeffs[i - j];
        }
    }

    for (std::size_t i = std::max(N, M); i < res.coeffs.size(); ++i)
    {
        for (std::size_t j = i - M; j <= N; ++j)
            res.coeffs[i] += p.coeffs[j]*q.coeffs[i - j];
    }

    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator*(const T a, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a*p.coeffs[i];
    return res;
}

template <typename T, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, order>
operator*(const Polynomial<T, order>& p, const T a) noexcept
{
    Polynomial<T, order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i]*a;
    return res;
}

template <typename T, std::size_t mon_order, std::size_t order>
[[nodiscard]] constexpr Polynomial<T, mon_order + order>
operator*(const Monomial<T, mon_order>& m, const Polynomial<T, order>& p) noexcept
{
    Polynomial<T, order + mon_order> res{};
    for (std::size_t i = mon_order; i < res.coeffs.size(); ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - mon_order];
    return res;
}

template <typename T, std::size_t order, std::size_t mon_order>
[[nodiscard]] constexpr Polynomial<T, order + mon_order>
operator*(const Polynomial<T, order>& p, const Monomial<T, mon_order>& m) noexcept
{
    Polynomial<T, order + mon_order> res{};
    for (std::size_t i = mon_order; i < res.coeffs.size(); ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - mon_order];
    return res;
}

template <typename T>
struct DynamicPolynomial
{
    using value_type = T;

    DynamicPolynomial() = default;
    DynamicPolynomial(std::size_t order):
        coeffs(order + 1) {}
    DynamicPolynomial(const std::span<T>& coeffs_):
        coeffs(coeffs_.size())
    {
        std::ranges::copy(coeffs_, coeffs.begin());
    }

    [[nodiscard]] constexpr bool operator==(const DynamicPolynomial& other) const noexcept = default;

    template<typename DomainType>
        requires std::integral<DomainType> || std::floating_point<DomainType>
    [[nodiscard]] value_type
    operator()(DomainType x) const noexcept
    {
        return detail::horner_eval<std::span<value_type>, DomainType, value_type>(std::span<value_type>(coeffs), x);
    }

    [[nodiscard]] DynamicPolynomial
    derivative() const noexcept
    {
        if (coeffs.size() < 2)
            return DynamicPolynomial(1UL);
        else
        {
            auto deriv = DynamicPolynomial(order());
            for (std::size_t i = 1; i < coeffs.size(); ++i)
                deriv.coeffs[i - 1] = value_type(i)*coeffs[i];
            return deriv;
        }
    }

    DynamicPolynomial&
    differentiate() noexcept
    {
        if (coeffs.size() >= 2)
        {
            for (std::size_t i = 0; i < coeffs.size() - 1; ++i)
                coeffs[i] = value_type(i)*coeffs[i + 1];
            coeffs.resize(coeffs.size() - 1);
        }
    }

    [[nodiscard]] std::size_t
    order() const noexcept { return coeffs.size() - std::min(1, coeffs.size()); }

    std::vector<T> coeffs;
};

} // namespace zdm
