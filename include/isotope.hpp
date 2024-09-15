#pragma once

#include <string_view>
#include "element.hpp"

class Isotope
{
public:
    Isotope() = default;
    Isotope(std::size_t atomic_number, std::size_t mass_number):
        m_atomic_number(atomic_number), m_mass_number(mass_number) {}

    [[nodiscard]] std::size_t atomic_number() const noexcept
    { return m_atomic_number; }

    [[nodiscard]] std::string_view symbol() const noexcept
    { return symbol(m_atomic_number); };

    [[nodiscard]] std::size_t mass_number() const noexcept
    { return m_mass_number; }

    [[nodiscard]] double mass() const noexcept
    { return 0.93149410242*m_mass_number; }

private:
    std::size_t m_mass_number;
    std::size_t m_atomic_number;
};