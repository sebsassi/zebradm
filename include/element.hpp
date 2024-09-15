#pragma once

#include <string_view>
#include <vector>
#include <array>
#include <span>

constexpr std::string_view symbol(std::size_t atomic_number)
{
    constexpr std::array<char[4], 119> periodic_table = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };

    return periodic_table[atomic_number];
};

constexpr std::span<std::pair<std::size_t, double>> natural_isotopes(std::size_t atomic_number)
{
    using OffsetSize = std::pair<std::size_t, std::size_t>;
    constexpr std::array<OffsetSize, 93> offset_size = {
        OffsetSize{0, 0}, 
        OffsetSize{1, 2}, // H
        OffsetSize{3, 2}, // He
        OffsetSize{5, 2}, // Li
        OffsetSize{7, 1}, // Be
        OffsetSize{8, 2}, // B
        OffsetSize{10, 2}, // C
        OffsetSize{12, 2}, // N
        OffsetSize{14, 3}, // O
        OffsetSize{17, 1}, // F
        OffsetSize{18, 3}, // Ne
    };

    if (atomic_number > 92) return std::span<std::pair<std::size_t, double>>{};
}

class Element
{
public:
    Element() = default;
    Element(std::size_t atomic_number): m_atomic_number(atomic_number) {}

    [[nodiscard]] std::size_t atomic_number() const noexcept
    { return m_atomic_number; }

    [[nodiscard]] std::string_view symbol() const noexcept
    { return symbol(m_atomic_number); };

private:
    std::size_t m_atomic_number;
};