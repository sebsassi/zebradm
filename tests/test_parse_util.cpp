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

#include <cassert>

#include "parse_util.hpp"

namespace
{

void test_find_next_non_whitespace()
{
    assert(zdm::detail::find_next_non_whitespace(""sv) == ""sv);
    assert(zdm::detail::find_next_non_whitespace("      "sv) == ""sv);
    assert(zdm::detail::find_next_non_whitespace(" hello"sv) == "hello"sv);
    assert(zdm::detail::find_next_non_whitespace(" \f\n\r\t\v"sv) == ""sv);
    assert(zdm::detail::find_next_non_whitespace(" \f\n\r\t\vhello"sv) == "hello"sv);
}

bool test_parse_unsigned_gives_correct_result(
    std::string_view input, std::uint64_t expected_number, std::string_view expected_output)
{
    const auto result = zdm::detail::parse_unsigned(input);
    if (!result.has_value()) return false;

    const auto& [number, output] = *result;
    return number == expected_number && output == expected_output;
}

void test_parse_unsigned()
{
    assert(zdm::detail::parse_unsigned(""sv) == std::unexpected(zdm::detail::GenericParseStatus::empty_string));
    assert(zdm::detail::parse_unsigned("hello"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));
    assert(zdm::detail::parse_unsigned("-1"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));
    assert(zdm::detail::parse_unsigned("+1"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));

    assert(test_parse_unsigned_gives_correct_result("237489"sv, 237489, ""sv));
    assert(test_parse_unsigned_gives_correct_result("0000237489"sv, 237489, ""sv));
    assert(test_parse_unsigned_gives_correct_result("237489 hello"sv, 237489, " hello"sv));
}

bool test_parse_signed_gives_correct_result(
    std::string_view input, std::int64_t expected_number, std::string_view expected_output)
{
    const auto result = zdm::detail::parse_signed(input);
    if (!result.has_value()) return false;

    const auto& [number, output] = *result;
    return number == expected_number && output == expected_output;
}

void test_parse_signed()
{
    assert(zdm::detail::parse_signed(""sv) == std::unexpected(zdm::detail::GenericParseStatus::empty_string));
    assert(zdm::detail::parse_signed("hello"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));
    assert(zdm::detail::parse_signed("-hello"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));
    assert(zdm::detail::parse_signed("+hello"sv) == std::unexpected(zdm::detail::GenericParseStatus::expected_digits));

    assert(test_parse_signed_gives_correct_result("237489", 237489, ""sv));
    assert(test_parse_signed_gives_correct_result("-237489", -237489, ""sv));
    assert(test_parse_signed_gives_correct_result("+237489", 237489, ""sv));
    assert(test_parse_signed_gives_correct_result("-237489 a", -237489, " a"sv));
}

}

int main()
{
    test_find_next_non_whitespace();
    test_parse_unsigned();
    test_parse_signed();
}
