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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace zdm
{

namespace detail
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

constexpr std::uint8_t atomic_number_of(std::string_view chemical_symbol)
{
    static constexpr std::array<std::uint8_t, 27UL*27UL> number_table = {
//      0  "a" "b"  "c"  "d"  "e" "f"  "g"  "h"  "i" "j""k" "l"  "m"  "n"  "o"  "p" "q""r"  "s"  "t"  "u" "v"  "w""x""y" "z"
        0,  0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // 0
        0,  0,  0,   89,  0,   0,  0,   47,  0,   0,  0, 0,  13,  95,  0,   0,   0,  0, 18,  33,  85,  79, 0,   0, 0, 0,  0,   // "A"
        5,  56, 0,   0,   0,   4,  0,   0,   107, 83, 0, 97, 0,   0,   0,   0,   0,  0, 35,  0,   0,   0,  0,   0, 0, 0,  0,   // "B"
        6,  20, 0,   0,   48,  58, 98,  0,   0,   0,  0, 0,  17,  96,  112, 27,  0,  0, 24,  55,  0,   29, 0,   0, 0, 0,  0,   // "C"
        0,  0,  105, 0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   110, 0,   0,  0,   0, 0, 66, 0,   // "D"
        0,  0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 68,  99,  0,   63, 0,   0, 0, 0,  0,   // "E"
        9,  0,  0,   0,   0,   26, 0,   0,   0,   0,  0, 0,  114, 100, 0,   0,   0,  0, 87,  0,   0,   0,  0,   0, 0, 0,  0,   // "F"
        0,  31, 0,   0,   64,  32, 0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "G"
        1,  0,  0,   0,   0,   2,  72,  80,  0,   0,  0, 0,  0,   0,   0,   67,  0,  0, 0,   108, 0,   0,  0,   0, 0, 0,  0,   // "H"
        53, 0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   49,  0,   0,  0, 77,  0,   0,   0,  0,   0, 0, 0,  0,   // "I"
        0,  0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "J"
        19, 0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 36,  0,   0,   0,  0,   0, 0, 0,  0,   // "K"
        0,  57, 0,   0,   0,   0,  0,   0,   0,   3,  0, 0,  0,   0,   0,   0,   0,  0, 103, 0,   0,   71, 116, 0, 0, 0,  0,   // "L"
        0,  0,  0,   115, 101, 0,  0,   12,  0,   0,  0, 0,  0,   0,   25,  42,  0,  0, 0,   0,   109, 0,  0,   0, 0, 0,  0,   // "M"
        7,  11, 41,  0,   60,  10, 0,   0,   113, 28, 0, 0,  0,   0,   0,   102, 93, 0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "N"
        8,  0,  0,   0,   0,   0,  0,   118, 0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   76,  0,   0,  0,   0, 0, 0,  0,   // "O"
        15, 91, 82,  0,   46,  0,  0,   0,   0,   0,  0, 0,  0,   61,  0,   84,  0,  0, 59,  0,   78,  94, 0,   0, 0, 0,  0,   // "P"
        0,  0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "Q"
        0,  88, 37,  0,   0,   75, 104, 111, 45,  0,  0, 0,  0,   0,   86,  0,   0,  0, 0,   0,   0,   44, 0,   0, 0, 0,  0,   // "R"
        16, 0,  51,  21,  0,   34, 0,   106, 0,   14, 0, 0,  0,   62,  50,  0,   0,  0, 38,  0,   0,   0,  0,   0, 0, 0,  0,   // "S"
        0,  73, 65,  43,  0,   52, 0,   0,   90,  22, 0, 0,  81,  69,  0,   0,   0,  0, 0,   117, 0,   0,  0,   0, 0, 0,  0,   // "T"
        92, 0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "U"
        23, 0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "V"
        74, 0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "W"
        0,  0,  0,   0,   0,   54, 0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "X"
        39, 0,  70,  0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   0,   0,   0,  0, 0,   0,   0,   0,  0,   0, 0, 0,  0,   // "Y"
        0,  0,  0,   0,   0,   0,  0,   0,   0,   0,  0, 0,  0,   0,   30,  0,   0,  0, 40,  0,   0,   0,  0,   0, 0, 0,  0,   // "Z"
    };
    auto first_index = std::size_t(std::max(0, chemical_symbol[0] - ('A' + 1)));
    auto second_index = std::size_t(std::max(0, chemical_symbol[1] - ('a' + 1)));
    first_index = (first_index < 27)*first_index;
    second_index = (second_index < 27)*second_index;

    return number_table[27*first_index + second_index];
}

constexpr std::string_view chemical_symbol_of(std::uint16_t atomic_number)
{
    static constexpr std::array<TinyString<2>, 119> periodic_table = {
        "\x00\x00",
        "H",                                                                                                                                                                                      "He",
        "Li", "Be",                                                                                                                                                 "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg",                                                                                                                                                 "Al", "Si", "P",  "S",  "Cl", "Ar",
        "K",  "Ca",                                                                                     "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr",                                                                                     "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };

    return periodic_table[atomic_number].view();
}

} // namespace detail

class Isotope
{
public:
    static constexpr double amu = 0.93149410372;

    constexpr Isotope() = default;

    constexpr Isotope(std::uint8_t atomic_number, std::uint16_t mass_number):
        m_mass_number(mass_number), m_atomic_number(atomic_number) {}

    [[nodiscard]] constexpr double
    mass() const noexcept { return double(m_mass_number)*amu; }

    [[nodiscard]] constexpr std::uint8_t
    atomic_number() const noexcept { return m_atomic_number; }

    [[nodiscard]] constexpr std::uint16_t
    mass_number() const noexcept { return m_mass_number; }

    [[nodiscard]] constexpr std::string_view
    symbol() const noexcept { return detail::chemical_symbol_of(m_atomic_number); }

private:
    std::uint16_t m_mass_number;
    std::uint8_t m_atomic_number;
};

class Element
{
public:
    static constexpr std::uint8_t max_atomic_number = 118;

    constexpr Element() = default;

    constexpr Element(std::uint8_t atomic_number):
        m_atomic_number(std::uint8_t(atomic_number < max_atomic_number)*atomic_number) {}

    constexpr Element(std::string_view chemical_symbol):
        m_atomic_number(detail::atomic_number_of(chemical_symbol)) {}

    [[nodiscard]] constexpr std::uint8_t
    atomic_number() const noexcept { return m_atomic_number; }

    [[nodiscard]] constexpr std::string_view
    symbol() const noexcept { return detail::chemical_symbol_of(m_atomic_number); }

    [[nodiscard]] constexpr std::span<const Isotope>
    primordial_isotopes() const noexcept
    {
        const auto offset_len = isotope_table_offsets_lengths[m_atomic_number];
        return std::span<const Isotope>(&isotope_tables[offset_len[0]], offset_len[1]);
    }

    [[nodiscard]] constexpr std::span<const double>
    isotope_abundances() const noexcept
    {
        const auto offset_len = isotope_table_offsets_lengths[m_atomic_number];
        return std::span<const double>(&isotope_abundance_tables[offset_len[0]], offset_len[2]);
    }

private:
    static constexpr std::array<std::array<std::uint8_t, 2>, 119> isotope_table_offsets_lengths = {
        std::array<std::uint8_t, 2>{0,   0 },    // 0
        std::array<std::uint8_t, 2>{1,   2 },    // 1   H
        std::array<std::uint8_t, 2>{3,   2 },    // 2   He
        std::array<std::uint8_t, 2>{5,   2 },    // 3   Li
        std::array<std::uint8_t, 2>{7,   1 },    // 4   Be
        std::array<std::uint8_t, 2>{8,   2 },    // 5   B
        std::array<std::uint8_t, 2>{10,  2 },    // 6   C
        std::array<std::uint8_t, 2>{12,  2 },    // 7   N
        std::array<std::uint8_t, 2>{14,  3 },    // 8   O
        std::array<std::uint8_t, 2>{17,  1 },    // 9   F
        std::array<std::uint8_t, 2>{18,  3 },    // 10  Ne
        std::array<std::uint8_t, 2>{21,  1 },    // 11  Na
        std::array<std::uint8_t, 2>{22,  3 },    // 12  Mg
        std::array<std::uint8_t, 2>{25,  1 },    // 13  As
        std::array<std::uint8_t, 2>{26,  3 },    // 14  Si
        std::array<std::uint8_t, 2>{29,  1 },    // 15  P
        std::array<std::uint8_t, 2>{30,  4 },    // 16  S
        std::array<std::uint8_t, 2>{34,  2 },    // 17  Cl
        std::array<std::uint8_t, 2>{36,  3 },    // 18  Ar
        std::array<std::uint8_t, 2>{39,  2 },    // 19  K
        std::array<std::uint8_t, 2>{41,  5 },    // 20  Ca
        std::array<std::uint8_t, 2>{46,  1 },    // 21  Sc
        std::array<std::uint8_t, 2>{47,  5 },    // 22  Ti
        std::array<std::uint8_t, 2>{52,  1 },    // 23  V
        std::array<std::uint8_t, 2>{53,  4 },    // 24  Cr
        std::array<std::uint8_t, 2>{57,  1 },    // 25  Mn
        std::array<std::uint8_t, 2>{58,  4 },    // 26  Fe
        std::array<std::uint8_t, 2>{62,  1 },    // 27  Co
        std::array<std::uint8_t, 2>{63,  5 },    // 28  Ni
        std::array<std::uint8_t, 2>{68,  2 },    // 29  Cu
        std::array<std::uint8_t, 2>{70,  5 },    // 30  Zn
        std::array<std::uint8_t, 2>{75,  2 },    // 31  Ga
        std::array<std::uint8_t, 2>{77,  3 },    // 32  Ge
        std::array<std::uint8_t, 2>{81,  1 },    // 33  As
        std::array<std::uint8_t, 2>{82,  5 },    // 34  Se
        std::array<std::uint8_t, 2>{87,  2 },    // 35  Br
        std::array<std::uint8_t, 2>{89,  5 },    // 36  Kr
        std::array<std::uint8_t, 2>{94,  1 },    // 37  Rb
        std::array<std::uint8_t, 2>{95,  4 },    // 38  Sr
        std::array<std::uint8_t, 2>{99,  1 },    // 39  Y
        std::array<std::uint8_t, 2>{100, 4 },    // 40  Zr
        std::array<std::uint8_t, 2>{104, 1 },    // 41  Nb
        std::array<std::uint8_t, 2>{105, 6 },    // 42  Mo
        std::array<std::uint8_t, 2>{0,   0 },    // 43  Tc
        std::array<std::uint8_t, 2>{111, 7 },    // 44  Ru
        std::array<std::uint8_t, 2>{118, 1 },    // 45  Rh
        std::array<std::uint8_t, 2>{119, 6 },    // 46  Pd
        std::array<std::uint8_t, 2>{125, 2 },    // 47  Ag
        std::array<std::uint8_t, 2>{127, 5 },    // 48  Cd
        std::array<std::uint8_t, 2>{133, 1 },    // 49  In
        std::array<std::uint8_t, 2>{134, 10},    // 50  Sn
        std::array<std::uint8_t, 2>{144, 2 },    // 51  Sb
        std::array<std::uint8_t, 2>{146, 6 },    // 52  Te
        std::array<std::uint8_t, 2>{152, 1 },    // 53  I
        std::array<std::uint8_t, 2>{153, 7 },    // 54  Xe
        std::array<std::uint8_t, 2>{160, 1 },    // 55  Cs
        std::array<std::uint8_t, 2>{161, 6 },    // 56  Ba
        std::array<std::uint8_t, 2>{167, 1 },    // 57  La
        std::array<std::uint8_t, 2>{168, 4 },    // 58  Ce
        std::array<std::uint8_t, 2>{172, 1 },    // 59  Pr
        std::array<std::uint8_t, 2>{173, 5 },    // 60  Nd
        std::array<std::uint8_t, 2>{0,   0 },    // 61  Pm
        std::array<std::uint8_t, 2>{178, 5 },    // 62  Sm
        std::array<std::uint8_t, 2>{183, 1 },    // 63  Eu
        std::array<std::uint8_t, 2>{184, 6 },    // 64  Gd
        std::array<std::uint8_t, 2>{190, 1 },    // 65  Tb
        std::array<std::uint8_t, 2>{191, 7 },    // 66  Dy
        std::array<std::uint8_t, 2>{198, 1 },    // 67  Ho
        std::array<std::uint8_t, 2>{199, 6 },    // 68  Er
        std::array<std::uint8_t, 2>{205, 1 },    // 69  Tm
        std::array<std::uint8_t, 2>{206, 7 },    // 70  Yb
        std::array<std::uint8_t, 2>{213, 1 },    // 71  Lu
        std::array<std::uint8_t, 2>{214, 5 },    // 72  Hf
        std::array<std::uint8_t, 2>{219, 2 },    // 73  Ta
        std::array<std::uint8_t, 2>{221, 4 },    // 74  W
        std::array<std::uint8_t, 2>{225, 1 },    // 75  Re
        std::array<std::uint8_t, 2>{226, 5 },    // 76  Os
        std::array<std::uint8_t, 2>{231, 2 },    // 77  Ir
        std::array<std::uint8_t, 2>{233, 5 },    // 78  Pt
        std::array<std::uint8_t, 2>{238, 1 },    // 79  Au
        std::array<std::uint8_t, 2>{239, 7 },    // 80  Hg
        std::array<std::uint8_t, 2>{246, 2 },    // 81  Tl
        std::array<std::uint8_t, 2>{248, 4 },    // 82  Pb
        std::array<std::uint8_t, 2>{252, 1 },    // 83  Bi
        std::array<std::uint8_t, 2>{0,   0 },    // 84  Po
        std::array<std::uint8_t, 2>{0,   0 },    // 85  At
        std::array<std::uint8_t, 2>{0,   0 },    // 86  Rn
        std::array<std::uint8_t, 2>{0,   0 },    // 87  Fr
        std::array<std::uint8_t, 2>{0,   0 },    // 88  Ra
        std::array<std::uint8_t, 2>{0,   0 },    // 89  Ac
        std::array<std::uint8_t, 2>{253, 0 },    // 90  Th
        std::array<std::uint8_t, 2>{0,   0 },    // 91  Pa
        std::array<std::uint8_t, 2>{254, 0 },    // 92  U
        std::array<std::uint8_t, 2>{0,   0 },    // 93  Np
        std::array<std::uint8_t, 2>{0,   0 },    // 94  Pu
        std::array<std::uint8_t, 2>{0,   0 },    // 95  Am
        std::array<std::uint8_t, 2>{0,   0 },    // 96  Cm
        std::array<std::uint8_t, 2>{0,   0 },    // 97  Bk
        std::array<std::uint8_t, 2>{0,   0 },    // 98  Cf
        std::array<std::uint8_t, 2>{0,   0 },    // 99  Es
        std::array<std::uint8_t, 2>{0,   0 },    // 100 Fm
        std::array<std::uint8_t, 2>{0,   0 },    // 101 Md
        std::array<std::uint8_t, 2>{0,   0 },    // 102 No
        std::array<std::uint8_t, 2>{0,   0 },    // 103 Lr
        std::array<std::uint8_t, 2>{0,   0 },    // 104 Rf
        std::array<std::uint8_t, 2>{0,   0 },    // 105 Db
        std::array<std::uint8_t, 2>{0,   0 },    // 106 Sg
        std::array<std::uint8_t, 2>{0,   0 },    // 107 Bh
        std::array<std::uint8_t, 2>{0,   0 },    // 108 Hs
        std::array<std::uint8_t, 2>{0,   0 },    // 109 Mt
        std::array<std::uint8_t, 2>{0,   0 },    // 110 Ds
        std::array<std::uint8_t, 2>{0,   0 },    // 111 Rg
        std::array<std::uint8_t, 2>{0,   0 },    // 112 Cn
        std::array<std::uint8_t, 2>{0,   0 },    // 113 Nh
        std::array<std::uint8_t, 2>{0,   0 },    // 114 Fl
        std::array<std::uint8_t, 2>{0,   0 },    // 115 Mc
        std::array<std::uint8_t, 2>{0,   0 },    // 116 Lv
        std::array<std::uint8_t, 2>{0,   0 },    // 117 Ts
        std::array<std::uint8_t, 2>{0,   0 },    // 118 Og
    };
    static constexpr std::array<Isotope, 287> isotope_tables = {
        Isotope{},                                                                                                                                                                          // 0        0
        Isotope{1, 1}, Isotope{1, 2},                                                                                                                                                       // 1   H    1
        Isotope{2, 3}, Isotope{2, 4},                                                                                                                                                       // 2   He   3
        Isotope{3, 6}, Isotope{3, 7},                                                                                                                                                       // 3   Li   5
        Isotope{4, 9},                                                                                                                                                                      // 4   Be   7
        Isotope{5, 10}, Isotope{5, 11},                                                                                                                                                     // 5   B    8
        Isotope{6, 12}, Isotope{6, 13},                                                                                                                                                     // 6   C    10
        Isotope{7, 14}, Isotope{7, 15},                                                                                                                                                     // 7   N    12
        Isotope{8, 16}, Isotope{8, 17}, Isotope{8, 18},                                                                                                                                     // 8   O    14
        Isotope{9, 19},                                                                                                                                                                     // 9   F    17
        Isotope{10, 20}, Isotope{10, 21}, Isotope{10, 22},                                                                                                                                  // 10  Ne   18
        Isotope{11, 23},                                                                                                                                                                    // 11  Na   21
        Isotope{12, 24}, Isotope{12, 25}, Isotope{12, 26},                                                                                                                                  // 12  Mg   22
        Isotope{13, 27},                                                                                                                                                                    // 13  Al   25
        Isotope{14, 28}, Isotope{14, 29}, Isotope{14, 30},                                                                                                                                  // 14  Si   26
        Isotope{15, 31},                                                                                                                                                                    // 15  P    29
        Isotope{16, 32}, Isotope{16, 33}, Isotope{16, 34}, Isotope{16, 36},                                                                                                                 // 16  S    30
        Isotope{17, 35}, Isotope{17, 37},                                                                                                                                                   // 17  Cl   34
        Isotope{18, 36}, Isotope{18, 38}, Isotope{18, 40},                                                                                                                                  // 18  Ar   36
        Isotope{19, 39}, Isotope{19, 40}, Isotope{19, 41},                                                                                                                                  // 19  K    39
        Isotope{20, 40}, Isotope{20, 42}, Isotope{20, 43}, Isotope{20, 44}, Isotope{20, 46}, Isotope{20, 48},                                                                               // 20  Ca   41
        Isotope{21, 45},                                                                                                                                                                    // 21  Sc   46
        Isotope{22, 46}, Isotope{22, 47}, Isotope{22, 48}, Isotope{22, 49}, Isotope{22, 50},                                                                                                // 22  Ti   47
        Isotope{23, 50}, Isotope{23, 51},                                                                                                                                                   // 23  V    52
        Isotope{24, 50}, Isotope{24, 52}, Isotope{24, 53}, Isotope{24, 54},                                                                                                                 // 24  Cr   53
        Isotope{25, 55},                                                                                                                                                                    // 25  Mn   57
        Isotope{26, 54}, Isotope{26, 56}, Isotope{26, 57}, Isotope{26, 58},                                                                                                                 // 26  Fe   58
        Isotope{27, 59},                                                                                                                                                                    // 27  Co   62
        Isotope{28, 58}, Isotope{28, 60}, Isotope{28, 61}, Isotope{28, 62}, Isotope{28, 64},                                                                                                // 28  Ni   63
        Isotope{29, 63}, Isotope{29, 65},                                                                                                                                                   // 29  Cu   68
        Isotope{30, 64}, Isotope{30, 66}, Isotope{30, 67}, Isotope{30, 68}, Isotope{30, 70},                                                                                                // 30  Zn   70
        Isotope{31, 69}, Isotope{31, 71},                                                                                                                                                   // 31  Ga   75
        Isotope{32, 70}, Isotope{32, 72}, Isotope{32, 73}, Isotope{32, 74}, Isotope{32, 76},                                                                                                // 32  Ge   77
        Isotope{33, 75},                                                                                                                                                                    // 33  As   81
        Isotope{34, 74}, Isotope{34, 76}, Isotope{34, 77}, Isotope{34, 78}, Isotope{34, 80}, Isotope{34, 82},                                                                               // 34  Se   82
        Isotope{35, 79}, Isotope{35, 81},                                                                                                                                                   // 35  Br   87
        Isotope{36, 78}, Isotope{36, 80}, Isotope{36, 82}, Isotope{36, 83}, Isotope{36, 84}, Isotope{36, 86},                                                                               // 36  Kr   89
        Isotope{37, 85}, Isotope{37, 87},                                                                                                                                                   // 37  Rb   94
        Isotope{38, 84}, Isotope{38, 86}, Isotope{38, 87}, Isotope{38, 88},                                                                                                                 // 38  Sr   95
        Isotope{39, 89},                                                                                                                                                                    // 39  Y    99
        Isotope{40, 90}, Isotope{40, 91}, Isotope{40, 92}, Isotope{40, 94}, Isotope{40, 96},                                                                                                // 40  Zr   100
        Isotope{41, 93},                                                                                                                                                                    // 41  Nb   104
        Isotope{42, 92}, Isotope{42, 94}, Isotope{42, 95}, Isotope{42, 96}, Isotope{42, 97}, Isotope{42, 98}, Isotope{42, 100},                                                             // 42  Mo   105
                                                                                                                                                                                            // 43  Tc
        Isotope{44, 96}, Isotope{44, 98}, Isotope{44, 99}, Isotope{44, 100}, Isotope{44, 101}, Isotope{44, 102}, Isotope{44, 104},                                                          // 44  Ru   111
        Isotope{45, 103},                                                                                                                                                                   // 45  Rh   118
        Isotope{46, 102}, Isotope{46, 104}, Isotope{46, 105}, Isotope{46, 106}, Isotope{46, 108}, Isotope{46, 110},                                                                         // 46  Pd   119
        Isotope{47, 107}, Isotope{47, 109},                                                                                                                                                 // 47  Ag   125
        Isotope{48, 106}, Isotope{48, 108}, Isotope{48, 110}, Isotope{48, 111}, Isotope{48, 112}, Isotope{48, 113}, Isotope{48, 114}, Isotope{48, 116},                                     // 48  Cd   127
        Isotope{49, 113}, Isotope{49, 115},                                                                                                                                                 // 49  In   133
        Isotope{50, 112}, Isotope{50, 114}, Isotope{50, 115}, Isotope{50, 116}, Isotope{50, 117}, Isotope{50, 118}, Isotope{50, 119}, Isotope{50, 120}, Isotope{50, 122}, Isotope{50, 124}, // 50  Sn   134
        Isotope{51, 121}, Isotope{51, 123},                                                                                                                                                 // 51  Sb   144
        Isotope{52, 120}, Isotope{52, 122}, Isotope{52, 123}, Isotope{52, 124}, Isotope{52, 125}, Isotope{52, 126}, Isotope{52, 128}, Isotope{52, 130},                                     // 52  Te   146
        Isotope{53, 127},                                                                                                                                                                   // 53  I    152
        Isotope{54, 124}, Isotope{54, 126}, Isotope{54, 128}, Isotope{54, 129}, Isotope{54, 130}, Isotope{54, 131}, Isotope{54, 132}, Isotope{54, 134}, Isotope{54, 136},                   // 54  Xe   153
        Isotope{55, 133},                                                                                                                                                                   // 55  Cs   160
        Isotope{56, 130}, Isotope{56, 132}, Isotope{56, 134}, Isotope{56, 135}, Isotope{56, 136}, Isotope{56, 137}, Isotope{56, 138},                                                       // 56  Ba   161
        Isotope{57, 138}, Isotope{57, 139},                                                                                                                                                 // 57  La   167
        Isotope{58, 136}, Isotope{58, 138}, Isotope{58, 140}, Isotope{58, 142},                                                                                                             // 58  Ce   168
        Isotope{59, 141},                                                                                                                                                                   // 59  Pr   172
        Isotope{60, 142}, Isotope{60, 143}, Isotope{60, 144}, Isotope{60, 145}, Isotope{60, 146}, Isotope{60, 148}, Isotope{60, 150},                                                       // 60  Nd   173
                                                                                                                                                                                            // 61  Pm
        Isotope{62, 144}, Isotope{62, 147}, Isotope{62, 148}, Isotope{62, 149}, Isotope{62, 150}, Isotope{62, 152}, Isotope{62, 154},                                                       // 62  Sm   178
        Isotope{63, 151}, Isotope{63, 153},                                                                                                                                                 // 63  Eu   183
        Isotope{64, 152}, Isotope{64, 154}, Isotope{64, 155}, Isotope{64, 156}, Isotope{64, 157}, Isotope{64, 158}, Isotope{64, 160},                                                       // 64  Gd   184
        Isotope{65, 159},                                                                                                                                                                   // 65  Tb   190
        Isotope{66, 156}, Isotope{66, 158}, Isotope{66, 160}, Isotope{66, 161}, Isotope{66, 162}, Isotope{66, 163}, Isotope{66, 164},                                                       // 66  Dy   191
        Isotope{67, 165},                                                                                                                                                                   // 67  Ho   198
        Isotope{68, 162}, Isotope{68, 164}, Isotope{68, 166}, Isotope{68, 167}, Isotope{68, 168}, Isotope{68, 170},                                                                         // 68  Er   199
        Isotope{69, 169},                                                                                                                                                                   // 69  Tm   205
        Isotope{70, 168}, Isotope{70, 170}, Isotope{70, 171}, Isotope{70, 172}, Isotope{70, 173}, Isotope{70, 174}, Isotope{70, 176},                                                       // 70  Yb   206
        Isotope{71, 175}, Isotope{71, 176},                                                                                                                                                 // 71  Lu   213
        Isotope{74, 174}, Isotope{72, 176}, Isotope{72, 177}, Isotope{72, 178}, Isotope{72, 179}, Isotope{72, 180},                                                                         // 72  Hf   214
        Isotope{73, 180}, Isotope{73, 181},                                                                                                                                                 // 73  Ta   219
        Isotope{74, 180}, Isotope{74, 182}, Isotope{74, 183}, Isotope{74, 184}, Isotope{74, 186},                                                                                           // 74  W    221
        Isotope{75, 185}, Isotope{75, 187},                                                                                                                                                 // 75  Re   225
        Isotope{76, 184}, Isotope{76, 186}, Isotope{76, 187}, Isotope{76, 188}, Isotope{76, 189}, Isotope{76, 190}, Isotope{76, 192},                                                       // 76  Os   226
        Isotope{77, 191}, Isotope{77, 193},                                                                                                                                                 // 77  Ir   231
        Isotope{78, 190}, Isotope{78, 192}, Isotope{78, 194}, Isotope{78, 195}, Isotope{78, 196}, Isotope{78, 198},                                                                         // 78  Pt   233
        Isotope{79, 197},                                                                                                                                                                   // 79  Au   238
        Isotope{80, 196}, Isotope{80, 198}, Isotope{80, 199}, Isotope{80, 200}, Isotope{80, 201}, Isotope{80, 202}, Isotope{80, 204},                                                       // 80  Hg   239
        Isotope{81, 203}, Isotope{81, 205},                                                                                                                                                 // 81  Tl   246
        Isotope{82, 204}, Isotope{82, 206}, Isotope{82, 207}, Isotope{82, 208},                                                                                                             // 82  Pb   248
        Isotope{83, 209},                                                                                                                                                                   // 83  Bi   252
                                                                                                                                                                                            // 84  Po
                                                                                                                                                                                            // 85  At
                                                                                                                                                                                            // 86  Rn
                                                                                                                                                                                            // 87  Fr
                                                                                                                                                                                            // 88  Ra
                                                                                                                                                                                            // 89  Ac
        Isotope{90, 232},                                                                                                                                                                   // 90  Th   253
                                                                                                                                                                                            // 91  Pa
        Isotope{92, 235}, Isotope{92, 238},                                                                                                                                                 // 92  U    254
                                                                                                                                                                                            // 93  Np
                                                                                                                                                                                            // 94  Pu
                                                                                                                                                                                            // 95  Am
                                                                                                                                                                                            // 96  Cm
                                                                                                                                                                                            // 97  Bk
                                                                                                                                                                                            // 98  Cf
                                                                                                                                                                                            // 99  Es
                                                                                                                                                                                            // 100 Fm
                                                                                                                                                                                            // 101 Md
                                                                                                                                                                                            // 102 No
                                                                                                                                                                                            // 103 Lr
                                                                                                                                                                                            // 104 Rf
                                                                                                                                                                                            // 105 Db
                                                                                                                                                                                            // 106 Sg
                                                                                                                                                                                            // 107 Bh
                                                                                                                                                                                            // 108 Hs
                                                                                                                                                                                            // 109 Mt
                                                                                                                                                                                            // 110 Ds
                                                                                                                                                                                            // 111 Rg
                                                                                                                                                                                            // 112 Cn
                                                                                                                                                                                            // 113 Nh
                                                                                                                                                                                            // 114 Fl
                                                                                                                                                                                            // 115 Mc
                                                                                                                                                                                            // 116 Lv
                                                                                                                                                                                            // 117 Ts
                                                                                                                                                                                            // 118 Og
    };

    static constexpr std::array<double, 287> isotope_abundance_tables = {
        0.0,
        0.999855, 0.000115,
        0.000002, 0.999998,
        0.048500, 0.951500,
        1.000000,
        0.199000, 0.801000,
        0.989000, 0.010600,
        0.996000, 0.003800,
        0.998000, 0.000384, 0.002050,
        1.000000,
        0.905000, 0.002700, 0.092500,
        1.000000,
        0.790000, 0.100000, 0.110000,
        1.000000,
        0.922000, 0.046700, 0.030700,
        1.000000,
        0.094800, 0.007600, 0.043700, 0.000200,
        0.758000, 0.242000,
        0.003340, 0.000630, 0.996000,
        0.933000, 0.000117, 0.067300,
        0.969000, 0.006470, 0.001350, 0.020900, 0.000040, 0.0018700,
        1.000000,
        0.082500, 0.074400, 0.737000, 0.054100, 0.051800,
        0.002500, 0.998000,
        0.043400, 0.838000, 0.095000, 0.023700,
        1.000000,
        0.058500, 0.918000, 0.021200, 0.002800,
        1.000000,
        0.681000, 0.262000, 0.011400, 0.036300, 0.009260,
        0.692000, 0.309000,
        0.492000, 0.277000, 0.040400, 0.184000, 0.006100,
        0.601000, 0.399000,
        0.205000, 0.274000, 0.077600, 0.365000, 0.077500,
        1.000000,
        0.008600, 0.092300, 0.076000, 0.237000, 0.498000, 0.088200,
        0.506000, 0.494000,
        0.003600, 0.022900, 0.116000, 0.115000, 0.570000, 0.173000,
        0.722000, 0.278000,
        0.005600, 0.098600, 0.070000, 0.826000,
        1.000000,
        0.515000, 0.112000, 0.171000, 0.174000, 0.028000,
        1.000000,
        0.147000, 0.091900, 0.159000, 0.167000, 0.095800, 0.243000, 0.097400,

        0.055400, 0.018700, 0.128000, 0.126000, 0.171000, 0.316000, 0.186000,
        1.000000,
        0.010200, 0.111000, 0.223000, 0.273000, 0.265000, 0.117000,
        0.518000, 0.482000,
        0.012500, 0.008900, 0.125000, 0.128000, 0.241000, 0.122000, 0.288000, 0.075100,
        0.042800, 0.957000,
        0.009700, 0.006600, 0.003400, 0.145000, 0.076800, 0.242000, 0.085900, 0.326000, 0.046300, 0.057900,
        0.572000, 0.428000,
        0.000900, 0.025500, 0.008900, 0.047400, 0.070700, 0.188000, 0.317000, 0.341000,
        1.000000,
        0.000950, 0.000890, 0.019100, 0.264000, 0.040700, 0.212000, 0.269000, 0.104000, 0.088600,
        1.000000,
        0.001100, 0.001000, 0.024200, 0.065900, 0.078500, 0.112000, 0.717000,
        0.000890, 0.999000,
        0.001860, 0.002510, 0.884000, 0.111000,
        1.000000,
        0.272000, 0.122000, 0.238000, 0.083000, 0.172000, 0.058000, 0.056000,

        0.030800, 0.150000, 0.113000, 0.138000, 0.073700, 0.267000, 0.227000,
        0.478000, 0.522000,
        0.002000, 0.021800, 0.148000, 0.205000, 0.157000, 0.248000, 0.219000,
        1.000000,
        0.000560, 0.000950, 0.023300, 0.189000, 0.255000, 0.249000, 0.283000,
        1.000000,
        0.001390, 0.016000, 0.335000, 0.229000, 0.270000, 0.149000,
        1.000000,
        0.001260, 0.030200, 0.142000, 0.218000, 0.161000, 0.319000, 0.129000,
        0.974000, 0.026000,
        0.001600, 0.052600, 0.186000, 0.273000, 0.136000, 0.351000,
        0.000120, 0.999880,
        0.001200, 0.265000, 0.143000, 0.306000, 0.284000,
        0.374000, 0.626000,
        0.000200, 0.015900, 0.019600, 0.132000, 0.161000, 0.263000, 0.408000,
        0.373000, 0.627000,
        0.000120, 0.007820, 0.329000, 0.338000, 0.252000, 0.073600,
        1.000000,
        0.001500, 0.100000, 0.169000, 0.231000, 0.132000, 0.297000, 0.068200,
        0.295000, 0.705000,
        0.014000, 0.241000, 0.221000, 0.524000,
        1.000000,






        0.000200, 1.000000,

        0.007200, 0.993000,

    };

    std::uint8_t m_atomic_number;
};

} // namespace zdm
