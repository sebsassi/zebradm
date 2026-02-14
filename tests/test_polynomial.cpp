/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include "polynomial.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace
{

bool test_horner_eval_evaluates_coeffs_order_0(
    const std::array<std::int64_t, 1>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0];
    return zdm::detail::horner_eval<std::array<std::int64_t, 1>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_1(
    const std::array<std::int64_t, 2>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 2>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_2(
    const std::array<std::int64_t, 3>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 3>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_3(
    const std::array<std::int64_t, 4>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 4>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_4(
    const std::array<std::int64_t, 5>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 5>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_5(
    const std::array<std::int64_t, 6>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 6>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_6(
    const std::array<std::int64_t, 7>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x + coeffs[6]*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 7>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_7(
    const std::array<std::int64_t, 8>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 8>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_8(
    const std::array<std::int64_t, 9>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x + coeffs[8]*x*x*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 9>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_9(
    const std::array<std::int64_t, 10>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x + coeffs[8]*x*x*x*x*x*x*x*x
            + coeffs[9]*x*x*x*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 10>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_0(
    const std::array<std::int64_t, 1>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0];
    return zdm::detail::estrin_eval<std::array<std::int64_t, 1>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_1(
    const std::array<std::int64_t, 2>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 2>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_2(
    const std::array<std::int64_t, 3>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 3>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_3(
    const std::array<std::int64_t, 4>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 4>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_4(
    const std::array<std::int64_t, 5>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 5>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_5(
    const std::array<std::int64_t, 6>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 6>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_6(
    const std::array<std::int64_t, 7>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 7>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_7(
    const std::array<std::int64_t, 8>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 8>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_8(
    const std::array<std::int64_t, 9>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x + coeffs[8]*x*x*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 9>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_9
(const std::array<std::int64_t, 10>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x
            + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x
            + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x + coeffs[8]*x*x*x*x*x*x*x*x
            + coeffs[9]*x*x*x*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 10>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

template <std::size_t N>
bool test_polynomial_derivative_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, std::max(N, 1UL) - 1UL>& expected_derivative)
{
    return polynomial.derivative() == expected_derivative;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_truncate_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, M>& expected_truncation)
{
    return polynomial.template truncate<M>() == expected_truncation;
}

template <std::size_t N>
bool test_polynomial_minus_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return -polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_addition_matches(
    const zdm::Polynomial<std::int64_t, N>& p, const zdm::Polynomial<std::int64_t, M>& q,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return p + q == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_right_addition_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, std::int64_t constant,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return polynomial + constant == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_left_addition_matches(
    std::int64_t constant, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return constant + polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_monomial_right_addition_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, zdm::Monomial<std::int64_t, M> monomial,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return polynomial + monomial == expected_result;
}

template <std::size_t M, std::size_t N>
bool test_polynomial_monomial_left_addition_matches(
    zdm::Monomial<std::int64_t, M> monomial, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return monomial + polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_subtraction_matches(
    const zdm::Polynomial<std::int64_t, N>& p, const zdm::Polynomial<std::int64_t, M>& q,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return p - q == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_right_subtraction_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, std::int64_t constant,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return polynomial - constant == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_left_subtraction_matches(
    std::int64_t constant, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return constant - polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_monomial_right_subtraction_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, zdm::Monomial<std::int64_t, M> monomial,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return polynomial - monomial == expected_result;
}

template <std::size_t M, std::size_t N>
bool test_polynomial_monomial_left_subtraction_matches(
    zdm::Monomial<std::int64_t, M> monomial, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, std::max(N, M)>& expected_result)
{
    return monomial - polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_multiplication_matches(
    const zdm::Polynomial<std::int64_t, N>& p, const zdm::Polynomial<std::int64_t, M>& q,
    const zdm::Polynomial<std::int64_t, N + M>& expected_result)
{
    return p*q == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_right_multiplication_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, std::int64_t constant,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return polynomial*constant == expected_result;
}

template <std::size_t N>
bool test_polynomial_constant_left_multiplication_matches(
    std::int64_t constant, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, N>& expected_result)
{
    return constant*polynomial == expected_result;
}

template <std::size_t N, std::size_t M>
bool test_polynomial_monomial_right_multiplication_matches(
    const zdm::Polynomial<std::int64_t, N>& polynomial, zdm::Monomial<std::int64_t, M> monomial,
    const zdm::Polynomial<std::int64_t, N + M>& expected_result)
{
    return polynomial*monomial == expected_result;
}

template <std::size_t M, std::size_t N>
bool test_polynomial_monomial_left_multiplication_matches(
    zdm::Monomial<std::int64_t, M> monomial, const zdm::Polynomial<std::int64_t, N>& polynomial,
    const zdm::Polynomial<std::int64_t, N + M>& expected_result)
{
    return monomial*polynomial == expected_result;
}

} // namespace

int main()
{
    assert(test_horner_eval_evaluates_coeffs_order_0(std::array<std::int64_t, 1>{1}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_0(std::array<std::int64_t, 1>{1}, 1));

    assert(test_horner_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, -1));

    assert(test_horner_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 2));

    assert(test_horner_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, -2));

    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 3));

    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -3));

    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -3));
    assert(test_horner_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 4));

    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -3));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 4));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -4));

    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -3));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 4));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -4));
    assert(test_horner_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 5));

    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -3));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 4));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -4));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 5));
    assert(test_horner_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -5));

    assert(test_estrin_eval_evaluates_coeffs_order_0(std::array<std::int64_t, 1>{1}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_0(std::array<std::int64_t, 1>{1}, 1));

    assert(test_estrin_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_1(std::array<std::int64_t, 2>{1, 2}, -1));

    assert(test_estrin_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_2(std::array<std::int64_t, 3>{1, 2, -3}, 2));

    assert(test_estrin_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_3(std::array<std::int64_t, 4>{1, 2, -3, -1}, -2));

    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_4(std::array<std::int64_t, 5>{1, 2, -3, -1, -4}, 3));

    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, 3));
    assert(test_estrin_eval_evaluates_coeffs_order_5(std::array<std::int64_t, 6>{1, 2, -3, -1, -4, 3}, -3));

    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 3));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, -3));
    assert(test_estrin_eval_evaluates_coeffs_order_6(std::array<std::int64_t, 7>{1, 2, -3, -1, -4, 3, -2}, 4));

    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 3));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -3));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, 4));
    assert(test_estrin_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2, 4}, -4));

    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 3));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -3));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 4));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, -4));
    assert(test_estrin_eval_evaluates_coeffs_order_8(std::array<std::int64_t, 9>{1, 2, -3, -1, -4, 3, -2, 4, -5}, 5));

    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 0));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 1));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -1));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 2));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -2));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 3));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -3));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 4));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -4));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, 5));
    assert(test_estrin_eval_evaluates_coeffs_order_9(std::array<std::int64_t, 10>{1, 2, -3, -1, -4, 3, -2, 4, -5, 5}, -5));

    assert(test_polynomial_derivative_matches(zdm::Polynomial<std::int64_t, 0>{}, zdm::Polynomial<std::int64_t, 0>{}));
    assert(test_polynomial_derivative_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{}));
    assert(test_polynomial_derivative_matches(zdm::Polynomial<std::int64_t, 1>{1, 1}, zdm::Polynomial<std::int64_t, 0>{1}));
    assert(test_polynomial_derivative_matches(zdm::Polynomial<std::int64_t, 2>{1, 1, 1}, zdm::Polynomial<std::int64_t, 1>{1, 2}));

    assert(test_polynomial_truncate_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{1}));
    assert(test_polynomial_truncate_matches(zdm::Polynomial<std::int64_t, 1>{1, 2}, zdm::Polynomial<std::int64_t, 0>{1}));
    assert(test_polynomial_truncate_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 2>{1, 2, 3}));

    assert(test_polynomial_minus_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{-1}));
    assert(test_polynomial_minus_matches(zdm::Polynomial<std::int64_t, 1>{1, 2}, zdm::Polynomial<std::int64_t, 1>{-1, -2}));
    assert(test_polynomial_minus_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 4>{-1, -2, -3, -4, -5}));

    assert(test_polynomial_addition_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{3}));
    assert(test_polynomial_addition_matches(zdm::Polynomial<std::int64_t, 1>{1, 2}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 1>{3, 2}));
    assert(test_polynomial_addition_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 1>{2, 2}, zdm::Polynomial<std::int64_t, 1>{3, 2}));
    assert(test_polynomial_addition_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 4>{5, 4, 3, 2, 1}, zdm::Polynomial<std::int64_t, 4>{6, 6, 6, 6, 6}));
    assert(test_polynomial_addition_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 2>{5, 4, 3}, zdm::Polynomial<std::int64_t, 4>{6, 6, 6, 4, 5}));

    assert(test_polynomial_constant_left_addition_matches(1, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{3}));
    assert(test_polynomial_constant_left_addition_matches(1, zdm::Polynomial<std::int64_t, 1>{2, 3}, zdm::Polynomial<std::int64_t, 1>{3, 3}));
    assert(test_polynomial_constant_left_addition_matches(1, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{3, 3, 4, 5, 6}));

    assert(test_polynomial_constant_right_addition_matches(zdm::Polynomial<std::int64_t, 0>{2}, 1, zdm::Polynomial<std::int64_t, 0>{3}));
    assert(test_polynomial_constant_right_addition_matches(zdm::Polynomial<std::int64_t, 1>{2, 3}, 1, zdm::Polynomial<std::int64_t, 1>{3, 3}));
    assert(test_polynomial_constant_right_addition_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, 1, zdm::Polynomial<std::int64_t, 4>{3, 3, 4, 5, 6}));

    assert(test_polynomial_monomial_right_addition_matches(zdm::Polynomial<std::int64_t, 0>{2}, zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{3}));
    assert(test_polynomial_monomial_right_addition_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 4>{3, 3, 4, 5, 6}));
    assert(test_polynomial_monomial_right_addition_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 2>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 5, 5, 6}));
    assert(test_polynomial_monomial_right_addition_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 4>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 7}));
    assert(test_polynomial_monomial_right_addition_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 6>{1}, zdm::Polynomial<std::int64_t, 6>{2, 3, 4, 5, 6, 0, 1}));

    assert(test_polynomial_monomial_left_addition_matches(zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{3}));
    assert(test_polynomial_monomial_left_addition_matches(zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{3, 3, 4, 5, 6}));
    assert(test_polynomial_monomial_left_addition_matches(zdm::Monomial<std::int64_t, 2>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{2, 3, 5, 5, 6}));
    assert(test_polynomial_monomial_left_addition_matches(zdm::Monomial<std::int64_t, 4>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 7}));
    assert(test_polynomial_monomial_left_addition_matches(zdm::Monomial<std::int64_t, 6>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 6>{2, 3, 4, 5, 6, 0, 1}));

    assert(test_polynomial_subtraction_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{-1}));
    assert(test_polynomial_subtraction_matches(zdm::Polynomial<std::int64_t, 1>{1, 2}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 1>{-1, 2}));
    assert(test_polynomial_subtraction_matches(zdm::Polynomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 1>{2, 2}, zdm::Polynomial<std::int64_t, 1>{-1, -2}));
    assert(test_polynomial_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 4>{5, 4, 3, 2, 1}, zdm::Polynomial<std::int64_t, 4>{-4, -2, 0, 2, 4}));
    assert(test_polynomial_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 2>{5, 4, 3}, zdm::Polynomial<std::int64_t, 4>{-4, -2, 0, 4, 5}));

    assert(test_polynomial_constant_left_subtraction_matches(1, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{-1}));
    assert(test_polynomial_constant_left_subtraction_matches(1, zdm::Polynomial<std::int64_t, 1>{2, 3}, zdm::Polynomial<std::int64_t, 1>{-1, 3}));
    assert(test_polynomial_constant_left_subtraction_matches(1, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{-1, 3, 4, 5, 6}));

    assert(test_polynomial_constant_right_subtraction_matches(zdm::Polynomial<std::int64_t, 0>{2}, 1, zdm::Polynomial<std::int64_t, 0>{1}));
    assert(test_polynomial_constant_right_subtraction_matches(zdm::Polynomial<std::int64_t, 1>{2, 3}, 1, zdm::Polynomial<std::int64_t, 1>{1, 3}));
    assert(test_polynomial_constant_right_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, 1, zdm::Polynomial<std::int64_t, 4>{1, 3, 4, 5, 6}));

    assert(test_polynomial_monomial_right_subtraction_matches(zdm::Polynomial<std::int64_t, 0>{2}, zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{1}));
    assert(test_polynomial_monomial_right_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 4>{1, 3, 4, 5, 6}));
    assert(test_polynomial_monomial_right_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 2>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 3, 5, 6}));
    assert(test_polynomial_monomial_right_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 4>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 5}));
    assert(test_polynomial_monomial_right_subtraction_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 6>{1}, zdm::Polynomial<std::int64_t, 6>{2, 3, 4, 5, 6, 0, -1}));

    assert(test_polynomial_monomial_left_subtraction_matches(zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{-1}));
    assert(test_polynomial_monomial_left_subtraction_matches(zdm::Monomial<std::int64_t, 0>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{-1, -3, -4, -5, -6}));
    assert(test_polynomial_monomial_left_subtraction_matches(zdm::Monomial<std::int64_t, 2>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{-2, -3, -3, -5, -6}));
    assert(test_polynomial_monomial_left_subtraction_matches(zdm::Monomial<std::int64_t, 4>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{-2, -3, -4, -5, -5}));
    assert(test_polynomial_monomial_left_subtraction_matches(zdm::Monomial<std::int64_t, 6>{1}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 6>{-2, -3, -4, -5, -6, 0, 1}));

    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{3}, zdm::Polynomial<std::int64_t, 0>{6}));
    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 4>{2, 4, 6, 8, 10}));
    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 4>{2, 4, 6, 8, 10}));
    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 1>{0, 2}, zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 5>{0, 2, 4, 6, 8, 10}));
    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{1, 2, 3, 4, 5}, zdm::Polynomial<std::int64_t, 1>{0, 2}, zdm::Polynomial<std::int64_t, 5>{0, 2, 4, 6, 8, 10}));
    assert(test_polynomial_multiplication_matches(zdm::Polynomial<std::int64_t, 1>{1, 2}, zdm::Polynomial<std::int64_t, 1>{3, 4}, zdm::Polynomial<std::int64_t, 2>{3, 10, 8}));

    assert(test_polynomial_constant_left_multiplication_matches(2, zdm::Polynomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{4}));
    assert(test_polynomial_constant_left_multiplication_matches(2, zdm::Polynomial<std::int64_t, 1>{2, 3}, zdm::Polynomial<std::int64_t, 1>{4, 6}));
    assert(test_polynomial_constant_left_multiplication_matches(2, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{4, 6, 8, 10, 12}));

    assert(test_polynomial_constant_right_multiplication_matches(zdm::Polynomial<std::int64_t, 0>{2}, 2, zdm::Polynomial<std::int64_t, 0>{4}));
    assert(test_polynomial_constant_right_multiplication_matches(zdm::Polynomial<std::int64_t, 1>{2, 3}, 2, zdm::Polynomial<std::int64_t, 1>{4, 6}));
    assert(test_polynomial_constant_right_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, 2, zdm::Polynomial<std::int64_t, 4>{4, 6, 8, 10, 12}));

    assert(test_polynomial_monomial_right_multiplication_matches(zdm::Polynomial<std::int64_t, 0>{2}, zdm::Monomial<std::int64_t, 0>{3}, zdm::Polynomial<std::int64_t, 0>{6}));
    assert(test_polynomial_monomial_right_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 4>{4, 6, 8, 10, 12}));
    assert(test_polynomial_monomial_right_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 2>{2}, zdm::Polynomial<std::int64_t, 6>{0, 0, 4, 6, 8, 10, 12}));
    assert(test_polynomial_monomial_right_multiplication_matches(zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Monomial<std::int64_t, 4>{2}, zdm::Polynomial<std::int64_t, 8>{0, 0, 0, 0, 4, 6, 8, 10, 12}));

    assert(test_polynomial_monomial_left_multiplication_matches(zdm::Monomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 0>{3}, zdm::Polynomial<std::int64_t, 0>{6}));
    assert(test_polynomial_monomial_left_multiplication_matches(zdm::Monomial<std::int64_t, 0>{2}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 4>{4, 6, 8, 10, 12}));
    assert(test_polynomial_monomial_left_multiplication_matches(zdm::Monomial<std::int64_t, 2>{2}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 6>{0, 0, 4, 6, 8, 10, 12}));
    assert(test_polynomial_monomial_left_multiplication_matches(zdm::Monomial<std::int64_t, 4>{2}, zdm::Polynomial<std::int64_t, 4>{2, 3, 4, 5, 6}, zdm::Polynomial<std::int64_t, 8>{0, 0, 0, 0, 4, 6, 8, 10, 12}));
}
