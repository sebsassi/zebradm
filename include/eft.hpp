#pragma once

#include "polynomial.hpp"
#include "linalg.hpp"

template <typename T>
struct EFTPair
{
    T nontransverse;
    T transverse;
};

template <typename T>
EFTPair<T> operator+(EFTPair<T> a, EFTPair<T> b)
{
    return {
        a.nontransverse + b.nontransverse,
        a.transverse + b. transverse
    };
}

template <typename T>
EFTPair<T> operator-(EFTPair<T> a, EFTPair<T> b)
{
    return {
        a.nontransverse - b.nontransverse,
        a.transverse - b. transverse
    };
}

template <typename T>
EFTPair<T> operator*(T a, EFTPair<T> b)
{
    return {a*b.nontransverse, a*b.transverse};
}

template <typename T>
EFTPair<T> operator*(EFTPair<T> a, T b)
{
    return {a.nontransverse*b, a.transverse*b};
}

template <std::size_t Order>
using NuclearFormFactor = Matrix<Polynomial<double, Order>, 2, 2>;

template <std::size_t Order>
using EFTFormFactor = Matrix<EFTPair<Polynomial<double, Order>>, 2, 2>;

template <std::size_t Order>
using RateFormFactor = Polynomial<EFTPair<double>, Order>;

using EFTCoeffs = Matrix<double, 11, 2>;

constexpr double spin_factor(double spin)
{
    return 0.75*spin*(spin + 1.0);
}

template <std::size_t Order>
EFTFormFactor<Order> form_factor_1_1(
    const NuclearFormFactor<Order>& ff_m)
{
    EFTFormFactor<Order> res{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
            res[i][j] = {ff_m[i][j], Polynomial<double, Order>{}};
    }
    return res;
}

template <std::size_t Order>
EFTFormFactor<Order> form_factor_3_3(
    double inv_ho_parameter,
    const NuclearFormFactor<Order>& ff_phipp,
    const NuclearFormFactor<Order>& ff_sigmap)
{
    constexpr std::size_t trunc_order = Order - std::min(2, Order);
    constexpr double nucleon_mass_sq = 0.0;
    const double phipp_coeff = (4.0/nucleon_mass_sq)*inv_ho_parameter;
    const double sigmap_coeff = 4.0*inv_ho_parameter;

    EFTFormFactor<Order> res{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
            res[i][j] = {
                Monomial<double, 2>{phipp_coeff}*ff_phipp[i][j].truncate<trunc_order>(),
                Monomial<double, 1>{sigmap_coeff}*ff_sigmap[i][j].truncate<trunc_order>()
            };
    }
}

template <std::size_t Order>
EFTFormFactor<Order> form_factor_4_4(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmapp,
    const NuclearFormFactor<Order>& ff_sigmap)
{
    const double prefactor = 0.06125*spin_factor(dm_spin);

    EFTFormFactor<Order> res{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
            res[i][j] = {
                prefactor*(ff_sigmapp[i][j] + ff_sigmap[i][j]),
                Polynomial<double, Order>{}
            };
    }
}

template <std::size_t Order>
EFTFormFactor<Order> form_factor_5_5(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmapp,
    const NuclearFormFactor<Order>& ff_sigmap)
{
    const double prefactor = spin_factor(dm_spin);
}

template <std::size_t Order>
EFTFormFactor<Order> form_factor_6_6(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmapp);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_7_7(
    const NuclearFormFactor<Order>& ff_sigmap);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_8_8(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_m,
    const NuclearFormFactor<Order>& ff_delta);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_9_9(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmap);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_10_10(
    const NuclearFormFactor<Order>& ff_sigmapp);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_11_11(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_m);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_1_3(
    const NuclearFormFactor<Order>& ff_m_phipp);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_4_5(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmap_delta);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_4_6(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmapp);

template <std::size_t Order>
EFTFormFactor<Order> form_factor_8_9(
    const double dm_spin,
    const NuclearFormFactor<Order>& ff_sigmap_delta);