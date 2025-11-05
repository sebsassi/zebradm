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
#include <concepts>
#include <ranges>
#include <span>
#include <vector>

namespace zdm
{

template <typename T, typename F>
concept vector_over = requires (T x, F a)
{
    x *= a;
    x /= a;
    { x*a } -> std::same_as<T>;
    { a*x } -> std::same_as<T>;
}
&& requires (T x, T y)
{
    x += y;
    x -= y;
    {x + y} -> std::same_as<T>;
    {x - y} -> std::same_as<T>;
};

template <typename T, typename S>
concept can_multiply = requires (T x, S y)
{
    { x*y } -> std::same_as<decltype(y*x)>;
};

template <typename FieldType, std::size_t Order>
struct Monomial
{
    using Field = FieldType;
    static constexpr std::size_t order = Order;
    FieldType coeff;
};

template <std::ranges::sized_range R, typename DomainType>
    requires can_multiply<std::ranges::range_value_t<R>, DomainType>
[[nodiscard]] constexpr auto
horner_eval(const R& coeffs, DomainType x) noexcept
{
    using FieldType = std::ranges::range_value_t<R>;
    using ResType = decltype((*std::begin(coeffs))*x);
    ResType res{};
    for (const FieldType& coeff : coeffs | std::views::reverse)
        res = coeff + res*x;
    return res;
}

template <typename FieldType, std::size_t Order>
struct Polynomial
{
    using Field = FieldType;
    static constexpr std::size_t order = Order;

    constexpr Polynomial() = default;
    explicit constexpr Polynomial(const std::array<FieldType, Order + 1>& coeffs_): coeffs(coeffs_) {}

    template <typename... Types>
        requires (std::convertible_to<Types, Field> && ...)
    constexpr Polynomial(Types... coeffs_): coeffs{FieldType(coeffs_)...} {}

    [[nodiscard]] constexpr bool operator==(const Polynomial& other) const noexcept = default;

    template <typename DomainType>
        requires vector_over<Field, DomainType> || vector_over<DomainType, Field>
    [[nodiscard]] constexpr FieldType
    operator()(DomainType x) const noexcept
    {
        if constexpr (order == 0)
            return coeffs[0];
        else
            return horner_eval(coeffs, x);

    }

    [[nodiscard]] constexpr auto
    derivative() const noexcept
    {
        if constexpr (order < 1)
            return Polynomial<FieldType, 0>();
        else if constexpr (order == 1)
            return Polynomial<FieldType, 0>(std::array<FieldType, 1>{coeffs[1]});
        else
        {
            Polynomial<FieldType, Order - 1> deriv;
            for (std::size_t i = 1; i <= Order; ++i)
            {
                if constexpr (std::integral<FieldType>
                        || std::floating_point<FieldType>)
                    deriv.coeffs[i - 1] = FieldType(i)*coeffs[i];
                else
                    deriv.coeffs[i - 1] = i*coeffs[i];
            }
            return deriv;
        }
    }

    template <std::size_t NewOrder>
        requires (NewOrder <= Order)
    [[nodiscard]] constexpr Polynomial<Field, NewOrder>
    truncate() const noexcept
    {
        Polynomial<FieldType, NewOrder> res{};
        for (std::size_t i = 1; i <= NewOrder; ++i)
            res.coeffs[i] = coeffs[i];
        return res;
    }

    std::array<FieldType, Order + 1> coeffs;
};

template <typename FieldType, typename... Types>
    requires (std::same_as<FieldType, Types> && ...)
Polynomial(FieldType, Types...) -> Polynomial<FieldType, sizeof...(Types)>;

template <typename FieldType, std::size_t Order>
[[nodiscard]] constexpr Polynomial<FieldType, Order>
operator-(const Polynomial<FieldType, Order>& p) noexcept
{
    Polynomial<FieldType, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = -p.coeffs[i];
    return res;
}

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::Field, P::order>>
        && std::same_as<Q, Polynomial<typename Q::Field, Q::order>>
[[nodiscard]] constexpr auto
operator+(const P& p, const Q& q) noexcept
{
    constexpr std::size_t min_order = std::min(P::order, Q::order);
    constexpr std::size_t max_order = std::max(P::order, Q::order);

    Polynomial<typename P::FieldType, max_order> res{};
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

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::Field, P::order>>
        && std::same_as<Q, Polynomial<typename Q::Field, Q::order>>
[[nodiscard]] constexpr auto
operator-(const P& p, const Q& q) noexcept
{
    constexpr std::size_t min_order = std::min(P::order, Q::order);
    constexpr std::size_t max_order = std::max(P::order, Q::order);

    Polynomial<typename P::FieldType, max_order> res{};
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

template <typename M, typename P>
    requires std::same_as<M, Monomial<typename M::Field, M::order>>
        && std::same_as<P, Polynomial<typename P::Field, P::order>>
[[nodiscard]] constexpr auto
operator*(const M& m, const P& p) noexcept
{
    Polynomial<typename M::FieldType, M::Order + P::order> res{};
    for (std::size_t i = M::order; i <= M::order + P::order; ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - M::order];
    return res;
}

template <typename P, typename M>
    requires std::same_as<P, Polynomial<typename P::Field, P::order>>
        && std::same_as<M, Monomial<typename M::Field, M::order>>
[[nodiscard]] constexpr auto
operator*(const P& p, const M& m) noexcept
{
    Polynomial<typename M::FieldType, M::Order + P::order> res{};
    for (std::size_t i = M::order; i <= M::order + P::order; ++i)
        res.coeffs[i] = m.coeff*p.coeffs[i - M::order];
}

template <typename P, typename Q>
    requires std::same_as<P, Polynomial<typename P::Field, P::order>>
        && std::same_as<Q, Polynomial<typename Q::Field, Q::order>>
[[nodiscard]] constexpr auto
operator*(const P& p, const Q& q) noexcept
{
    Polynomial<typename P::FieldType, P::Order + Q::order> res{};
    /* TODO */
}

template <typename FieldType, std::size_t Order>
[[nodiscard]] constexpr Polynomial<FieldType, Order>
operator*(const FieldType a, const Polynomial<FieldType, Order>& p) noexcept
{
    Polynomial<FieldType, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = a*p.coeffs[i];
    return res;
}

template <typename FieldType, std::size_t Order>
[[nodiscard]] constexpr Polynomial<FieldType, Order>
operator*(const Polynomial<FieldType, Order>& p, const FieldType a) noexcept
{
    Polynomial<FieldType, Order> res{};
    for (std::size_t i = 0; i < res.coeffs.size(); ++i)
        res.coeffs[i] = p.coeffs[i]*a;
    return res;
}

template <typename FieldType, std::size_t Order, std::size_t TruncOrder>
[[nodiscard]] constexpr Polynomial<FieldType, TruncOrder>
truncate(const Polynomial<FieldType, Order>& p)
{
    /* TODO */
}

template <typename FieldType>
struct DynamicPolynomial
{
    DynamicPolynomial() = default;
    DynamicPolynomial(std::size_t size):
        coeffs(size) {}
    DynamicPolynomial(const std::span<FieldType>& coeffs_):
        coeffs(coeffs_.size())
    {
        std::ranges::copy(coeffs_, coeffs.begin());
    }

    [[nodiscard]] constexpr bool operator==(const DynamicPolynomial& other) const noexcept = default;

    template<typename DomainType>
        requires std::integral<DomainType> || std::floating_point<DomainType>
    [[nodiscard]] FieldType
    operator()(DomainType x) const noexcept
    {
        FieldType res{};
        for (const FieldType& coeff : coeffs | std::views::reverse)
            res = coeff + res*x;
        return res;
    }

    [[nodiscard]] auto
    derivative() const noexcept
    {
        if (coeffs.size() < 2)
            return DynamicPolynomial<FieldType>(1UL);
        else if (coeffs.size() >= 2)
        {
            DynamicPolynomial<FieldType> deriv(coeffs.size() - 1);
            for (std::size_t i = 1; i < coeffs.size(); ++i)
            {
                if constexpr (std::integral<FieldType> || std::floating_point<FieldType>)
                    deriv.coeffs[i - 1] = FieldType(i)*coeffs[i];
                else
                    deriv.coeffs[i - 1] = i*coeffs[i];
            }
            return deriv;
        }
    }

    [[nodiscard]] std::size_t
    order() const noexcept { return coeffs.size() - std::min(1, coeffs.size()); }

    std::vector<FieldType> coeffs;
};

} // namespace zdm
