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

#include <cassert>
#include <cstdint>

namespace
{

bool test_horner_eval_evaluates_coeffs_order_0(const std::array<std::int64_t, 1>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0];
    return zdm::detail::horner_eval<std::array<std::int64_t, 1>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_1(const std::array<std::int64_t, 2>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 2>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_2(const std::array<std::int64_t, 3>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 3>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_3(const std::array<std::int64_t, 4>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 4>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_4(const std::array<std::int64_t, 5>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 5>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_5(const std::array<std::int64_t, 6>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 6>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_6(const std::array<std::int64_t, 7>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x + coeffs[6]*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 7>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_horner_eval_evaluates_coeffs_order_7(const std::array<std::int64_t, 8>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x;
    return zdm::detail::horner_eval<std::array<std::int64_t, 8>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_0(const std::array<std::int64_t, 1>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0];
    return zdm::detail::estrin_eval<std::array<std::int64_t, 1>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_1(const std::array<std::int64_t, 2>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 2>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_2(const std::array<std::int64_t, 3>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 3>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_3(const std::array<std::int64_t, 4>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 4>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_4(const std::array<std::int64_t, 5>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 5>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_5(const std::array<std::int64_t, 6>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 6>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_6(const std::array<std::int64_t, 7>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x + coeffs[6]*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 7>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
}

bool test_estrin_eval_evaluates_coeffs_order_7(const std::array<std::int64_t, 8>& coeffs, std::int64_t x)
{
    const std::int64_t expected_result = coeffs[0] + coeffs[1]*x + coeffs[2]*x*x + coeffs[3]*x*x*x + coeffs[4]*x*x*x*x + coeffs[5]*x*x*x*x*x + coeffs[6]*x*x*x*x*x*x + coeffs[7]*x*x*x*x*x*x*x;
    return zdm::detail::estrin_eval<std::array<std::int64_t, 8>, std::int64_t, std::int64_t>(coeffs, x) == expected_result;
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

    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, 0));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, 1));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, -1));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, 2));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, -2));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, 3));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, -3));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, 4));
    assert(test_horner_eval_evaluates_coeffs_order_7(std::array<std::int64_t, 8>{1, 2, -3, -1, -4, 3, -2}, -4));

}
