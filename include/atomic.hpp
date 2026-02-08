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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace zdm
{

template <std::size_t N>
class TinyString
{
public:
    constexpr TinyString() = default;
    constexpr TinyString(std::string_view string):
        m_data{string[0], string[1]} {}
    constexpr TinyString(const char* string):
        m_data{string[0], string[1]} {}

    [[nodiscard]] constexpr operator std::string_view() noexcept { return std::string_view(m_data.data(), m_data.size()); }

    [[nodiscard]] constexpr std::string_view view() noexcept { return std::string_view(m_data.data(), m_data.size()); }

private:
    std::array<char, N> m_data;
};

struct Isotope
{
    static constexpr double amu = 0.93149410372;

    std::uint16_t atomic_number;
    std::uint16_t mass_number;

    [[nodiscard]] constexpr double
    mass() const noexcept { return double(mass_number)*amu; }
};

struct Element
{
    TinyString<2> symbol;
    std::uint16_t atomic_number;
};

} // namespace zdm
