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

    [[nodiscard]] constexpr std::string_view view() const noexcept { return std::string_view(m_data.data(), m_data.size()); }

private:
    std::array<char, N> m_data;
};

constexpr std::string_view chemical_symbol_of(std::uint16_t atomic_number)
{
    static constexpr std::array<TinyString<2>, 118> periodic_table = {
        "H", "He",
        "Li", "Be",                                                                                                                                                 "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg",                                                                                                                                                 "Al", "Si", "P",  "S",  "Cl", "Ar",
        "K",  "Ca",                                                                                     "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr",                                                                                     "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };

    return periodic_table[atomic_number].view();
}

struct Isotope
{
    static constexpr double amu = 0.93149410372;

    std::uint16_t atomic_number;
    std::uint16_t mass_number;

    [[nodiscard]] constexpr double
    mass() const noexcept { return double(mass_number)*amu; }

    [[nodiscard]] constexpr std::string_view symbol() const { return chemical_symbol_of(atomic_number); }
};

struct Element
{
    std::uint16_t atomic_number;

    [[nodiscard]] constexpr std::string_view symbol() const { return chemical_symbol_of(atomic_number); }
};

} // namespace zdm
