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
horner_eval(const R& coeffs, DomainType x) noexcept
{
    constexpr auto size = std::tuple_size_v<R>;

    if constexpr (size == 0)
        return ResultType{};
    else
    {
        ResultType res{};
        for (std::size_t i = size; i > 0; --i)
            res = coeffs[i - 1] + res*x;
        return res;
    }
}

template <typename R, typename DomainType, typename ResultType>
    requires fma_operable<std::ranges::range_value_t<R>, ResultType, DomainType>
[[nodiscard]] constexpr auto
horner_eval(const R& coeffs, DomainType x) noexcept
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

template <typename T, std::size_t Order>
struct Monomial
{
    using value_type = T;

    static constexpr std::size_t order = Order;

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
            return Monomial<value_type, 0>(coeff);
        else
            return Monomial<value_type, order - 1>(value_type(order)*coeff);
    }

    template <std::size_t NewOrder>
        requires (NewOrder <= order)
    [[nodiscard]] constexpr Monomial<value_type, NewOrder>
    truncate() const noexcept
    {
        Monomial<value_type, NewOrder> res{};
        if constexpr (NewOrder == order)
            return *this;
        else
            return Monomial<value_type, NewOrder>{};
    }
};

template <typename T, std::size_t Order>
struct Polynomial
{
    using value_type = T;
    static constexpr std::size_t order = Order;

    std::array<value_type, Order + 1> coeffs;

    constexpr Polynomial() = default;
    explicit constexpr Polynomial(const std::array<value_type, Order + 1>& coeffs_): coeffs(coeffs_) {}

    template <typename... Types>
        requires (std::convertible_to<Types, value_type> && ...)
    constexpr Polynomial(Types... coeffs_): coeffs{value_type(coeffs_)...} {}

    [[nodiscard]] constexpr bool operator==(const Polynomial& other) const noexcept = default;

    template <subfield_of<value_type> DomainType>
    [[nodiscard]] constexpr value_type
    operator()(DomainType x) const noexcept
    {
        if constexpr (order == 0)
            return coeffs[0];
        else
            return detail::horner_eval<const std::array<value_type, order + 1>&, DomainType, value_type>(coeffs, x);
    }

    [[nodiscard]] constexpr auto
    derivative() const noexcept requires zdm::arithmetic<value_type>
    {
        if constexpr (order < 1)
            return Polynomial<T, 0>();
        else if constexpr (order == 1)
            return Polynomial<T, 0>(std::array<T, 1>{coeffs[1]});
        else
        {
            Polynomial<T, Order - 1> deriv;
            for (std::size_t i = 1; i <= Order; ++i)
                deriv.coeffs[i - 1] = T(i)*coeffs[i];
            return deriv;
        }
    }

    template <std::size_t NewOrder>
        requires (NewOrder <= Order)
    [[nodiscard]] constexpr Polynomial<value_type, NewOrder>
    truncate() const noexcept
    {
        Polynomial<T, NewOrder> res{};
        for (std::size_t i = 1; i <= NewOrder; ++i)
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

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::value_type, P::order>>
        && std::same_as<Q, Polynomial<typename Q::value_type, Q::order>>
[[nodiscard]] constexpr auto
operator+(const P& p, const Q& q) noexcept
{
    constexpr std::size_t min_order = std::min(P::order, Q::order);
    constexpr std::size_t max_order = std::max(P::order, Q::order);

    Polynomial<typename P::value_type, max_order> res{};
    for (std::size_t i = 0; i <= min_order; ++i)
        res.coeffs[i] = p.coeffs[i] + q.coeffs[i];

    for (std::size_t i = min_order + 1; i <= max_order; ++i)
    {
        if constexpr (P::order > Q::order)
            res.coeffs[i] = p.coeffs[i];
        else
            res.coeffs[i] = q.coeffs[i];
    }

    return res;
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator+(const T a, const Polynomial<T, Order>& p) noexcept
{
    Polynomial<T, Order> res{};
    res.coeffs[0] = a + p.coeffs[0];
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a + p.coeffs[i];
    return res;
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator+(const Polynomial<T, Order>& p, const T a) noexcept
{
    Polynomial<T, Order> res{};
    res.coeffs[0] = p.coeffs[0] + a;
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a + p.coeffs[i];
    return res;
}

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::value_type, P::order>>
        && std::same_as<Q, Polynomial<typename Q::value_type, Q::order>>
[[nodiscard]] constexpr auto
operator-(const P& p, const Q& q) noexcept
{
    constexpr std::size_t min_order = std::min(P::order, Q::order);
    constexpr std::size_t max_order = std::max(P::order, Q::order);

    Polynomial<typename P::value_type, max_order> res{};
    for (std::size_t i = 0; i <= min_order; ++i)
        res.coeffs[i] = p.coeffs[i] - q.coeffs[i];
 
    for (std::size_t i = min_order + 1; i <= max_order; ++i)
    {
        if constexpr (P::order > Q::order)
            res.coeffs[i] = p.coeffs[i];
        else
            res.coeffs[i] = -q.coeffs[i];
    }

    return res;
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator-(const T a, const Polynomial<T, Order>& p) noexcept
{
    Polynomial<T, Order> res{};
    res.coeffs[0] = a - p.coeffs[0];
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a - p.coeffs[i];
    return res;
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator-(const Polynomial<T, Order>& p, const T a) noexcept
{
    Polynomial<T, Order> res{};
    res.coeffs[0] = p.coeffs[0] - a;
    for (std::size_t i = 1; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a - p.coeffs[i];
    return res;
}

template <typename M, typename P>
    requires std::same_as<M, Monomial<typename M::value_type, M::order>>
        && std::same_as<P, Polynomial<typename P::value_type, P::order>>
[[nodiscard]] constexpr auto
operator*(const M& m, const P& p) noexcept
{
    Polynomial<typename M::value_type, M::order + P::order> res{};
    for (std::size_t i = M::order; i <= M::order + P::order; ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - M::order];
    return res;
}

template <typename P, typename M>
    requires std::same_as<P, Polynomial<typename P::value_type, P::order>>
        && std::same_as<M, Monomial<typename M::value_type, M::order>>
[[nodiscard]] constexpr auto
operator*(const P& p, const M& m) noexcept
{
    Polynomial<typename M::value_type, M::order + P::order> res{};
    for (std::size_t i = M::order; i <= M::order + P::order; ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - M::order];
}

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::value_type, P::order>>
        && std::same_as<Q, Polynomial<typename Q::value_type, Q::order>>
        && std::same_as<typename P::value_type, typename Q::value_type>
[[nodiscard]] constexpr auto
operator*(const P& p, const Q& q) noexcept
{
    Polynomial<typename P::Field, P::order + Q::order> res{};
    for (std::size_t i = 0; i < P::order + Q::order; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
            res[i] += p[j]*q[i - j];
    }
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator*(const T a, const Polynomial<T, Order>& p) noexcept
{
    Polynomial<T, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a*p.coeffs[i];
    return res;
}

template <typename T, std::size_t Order>
[[nodiscard]] constexpr Polynomial<T, Order>
operator*(const Polynomial<T, Order>& p, const T a) noexcept
{
    Polynomial<T, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i]*a;
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

    [[nodiscard]] auto
    derivative() const noexcept
    {
        if (coeffs.size() < 2)
            return DynamicPolynomial<value_type>(1UL);
        else if (coeffs.size() >= 2)
        {
            DynamicPolynomial<value_type> deriv(coeffs.size() - 1);
            for (std::size_t i = 1; i < coeffs.size(); ++i)
                deriv.coeffs[i - 1] = value_type(i)*coeffs[i];
            return deriv;
        }
    }

    [[nodiscard]] std::size_t
    order() const noexcept { return coeffs.size() - std::min(1, coeffs.size()); }

    std::vector<T> coeffs;
};

} // namespace zdm
