#pragma once

#include <array>

#include "concepts.hpp"
#include "linalg.hpp"

namespace zdm::la
{

template <arithmetic T, std::size_t N>
struct Vector
{
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using iterator = std::array<T, N>::iterator;
    using const_iterator = std::array<T, N>::const_iterator;

    std::array<T, N> array;

    constexpr Vector() = default;
    constexpr Vector(const std::array<T, N>& arr): array{arr} {}

    template <arithmetic... Types>
    constexpr Vector(Types... values): array{T(values)...} {};

    [[nodiscard]] constexpr
    operator std::array<T, N>() const noexcept { return array; }

    [[nodiscard]] constexpr auto operator<=>(const Vector& other) const noexcept = default;

    [[nodiscard]] constexpr reference
    operator[](size_type i) noexcept { return array[i]; }

    [[nodiscard]] constexpr const_reference
    operator[](size_type i) const noexcept { return array[i]; }

    [[nodiscard]] constexpr reference
    at(size_type i) noexcept { return array.at(i); }

    [[nodiscard]] constexpr const_reference
    at(size_type i) const noexcept { return array.at(i); }

    [[nodiscard]] constexpr reference
    front() noexcept { return array.front(); }

    [[nodiscard]] constexpr const_reference
    front() const noexcept { return array.front(); }

    [[nodiscard]] constexpr reference
    back() noexcept { return array.back(); }

    [[nodiscard]] constexpr const_reference
    back() const noexcept { return array.back(); }

    [[nodiscard]] constexpr pointer
    data() noexcept { return array.data(); }

    [[nodiscard]] constexpr const_pointer
    data() const noexcept { return array.data(); }

    [[nodiscard]] constexpr iterator
    begin() noexcept { return array.begin(); }

    [[nodiscard]] constexpr const_iterator
    begin() const noexcept { return array.begin(); }

    [[nodiscard]] constexpr const_iterator
    cbegin() const noexcept { return array.cbegin(); }

    [[nodiscard]] constexpr iterator
    end() noexcept { return array.end(); }

    [[nodiscard]] constexpr const_iterator
    end() const noexcept { return array.end(); }

    [[nodiscard]] constexpr const_iterator
    cend() const noexcept { return array.cend(); }

    [[nodiscard]] constexpr bool
    empty() const noexcept { return array.empty(); }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return array.size(); }

    [[nodiscard]] constexpr size_type
    max_size() const noexcept { return array.max_size(); }

    constexpr void
    fill(const value_type& value) noexcept { array.fill(value); }

    constexpr void
    swap(Vector& other) noexcept { array.swap(other.array); }

    [[nodiscard]] friend constexpr Vector
    operator+(const Vector& a, const Vector& b) noexcept { return add(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator+(const Vector& a, const value_type& b) noexcept { return add(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator+(const value_type& a, const Vector& b) noexcept { return add(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator-(const Vector& a) noexcept { return minus(a); }

    [[nodiscard]] friend constexpr Vector
    operator-(const Vector& a, const Vector& b) noexcept { return sub(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator-(const Vector& a, const value_type& b) noexcept { return sub(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator-(const value_type& a, const Vector& b) noexcept { return sub(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator*(const value_type& a, const Vector& b) noexcept { return mul(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator*(const Vector& a, const value_type& b) noexcept { return mul(a, b); }

    [[nodiscard]] friend constexpr Vector
    operator/(const Vector& a, const value_type& b) noexcept { return div(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator+=(Vector& a, const Vector& b) noexcept { return add_assign(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator+=(Vector& a, const value_type& b) noexcept { return add_assign(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator-=(Vector& a, const Vector& b) noexcept { return sub_assign(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator-=(Vector& a, const value_type& b) noexcept { return sub_assign(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator*=(Vector& a, const value_type& b) noexcept { return mul_assign(a, b); }

    [[nodiscard]] friend constexpr Vector&
    operator/=(Vector& a, const value_type& b) noexcept { return div_assign(a, b); }
};

template <arithmetic T, typename... Ts>
    requires (std::same_as<T, Ts> && ...)
Vector(T, Ts...) -> Vector<T, sizeof...(Ts) + 1>;

} // namespace zdm::la

namespace std
{

template <zdm::arithmetic T, std::size_t N>
struct tuple_size<zdm::la::Vector<T, N>>: std::integral_constant<std::size_t, N> {};

template <std::size_t I, zdm::arithmetic T, std::size_t N>
struct tuple_element<I, zdm::la::Vector<T, N>>
{
    using type = T;
};

} // namespace std
