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
#pragma once

namespace zdm::detail
{

enum class GenericParseStatus: std::uint8_t
{
    success,
    empty_string,
    expected_digits,
};

[[nodiscard]] constexpr std::string_view
find_next_non_whitespace(std::string_view str) noexcept
{
    if (str.size() == 0) return str;
    std::size_t index = 0;
    while (index < str.size() && std::isspace(str[index])) ++index;
    return str.substr(index);
}

[[nodiscard]] constexpr std::expected<std::pair<std::uint64_t, std::string_view>, GenericParseStatus>
parse_unsigned(std::string_view str) noexcept
{
    if (str.size() == 0)
        return std::unexpected(GenericParseStatus::empty_string);
    if (str[0] < '0' || '9' < str[0])
        return std::unexpected(GenericParseStatus::expected_digits);

    auto value = std::uint64_t(str[0] - '0');
    std::size_t length = 1;
    for (; length < str.size() && '0' <= str[length] && str[length] <= '9'; ++length)
        value = value*10 + std::uint64_t(str[length] - '0');

    return std::pair{value, str.substr(length)};
}

[[nodiscard]] constexpr std::expected<std::pair<std::int64_t, std::string_view>, GenericParseStatus>
parse_signed(std::string_view str) noexcept
{
    if (str.size() == 0)
        return std::unexpected(GenericParseStatus::empty_string);
    std::int64_t sign = 1;
    if (str[0] == '-')
    {
        sign = -1;
        str = str.substr(1);
    }
    if (str[0] == '+')
        str = str.substr(1);

    return parse_unsigned(str)
        .transform([&](const auto& value) {
            return std::pair{sign*std::int64_t(value.first), value.second};
        });
}

};
