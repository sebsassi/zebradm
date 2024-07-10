#pragma once

#include <vector>
#include <array>
#include <span>
#include <cmath>
#include <ranges>

#define NI 2
#define N_EFT 11
#define KMAX 5
#define NQ 3
#define N_FF_CFF 4

template <typename T, std::size_t... Extents>
struct MultiArray
{
    explicit constexpr MultiArray(std::array<double, (Extents * ...)>& data_)
    : data(data_) {}

    constexpr T& operator()(std::size_t multi_index...)
        requires { sizeof...(multi_index) == ndim(); }
    {
        constexpr std::array<std::size_t, ndim()> strides_ = strides();

        std::size_t index = 0;
        std::size_t i = 0;
        ((index += multi_index*strides_[i++]), ...);
        return index;
    }

    constexpr std::size_t size() const { return (Extents * ...); }
    constexpr std::size_t ndim() const { return sizeof...(Extents); }
    constexpr std::array<std::size_t, ndim()> extents() const { return {Extents...}; }

    constexpr std::array<std::size_t, ndim()> strides() const
    {
        const ext = extents();
        std::array<std::size_t, ndim()> res{};
        res[ndim() - 1] = 1;
        for (std::size_t i = 2; i <= ndim(); ++i)
        {
            res[ndim() - i] = res[ndim() + 1 - i]*ext[ndim() + 1 - i];
        }
        return res;
    }

    std::array<double, (Extents * ...)> data;
};

constexpr std::size_t index_erc(const std::array<std::size_t, 4>& multi_index)
{
    return NQ*(KMAX*(N_EFT*multi_index[0] + multi_index[1]) + multi_index[2]) + multi_index[3];
}

constexpr std::array<double, NI*N_EFT*KMAX*NQ> eft_response_coeff()
{
    const std::array<std::array<std::size_t, 4>, 12> unity_indices = {
        std::array<std::size_t, 4>{0,0,0,0},
        std::array<std::size_t, 4>{0,2,1,2},
        std::array<std::size_t, 4>{0,3,2,0},
        std::array<std::size_t, 4>{0,3,3,0},
        std::array<std::size_t, 4>{1,4,0,1},
        std::array<std::size_t, 4>{0,4,4,2},
        std::array<std::size_t, 4>{0,5,2,2},
        std::array<std::size_t, 4>{1,7,0,0},
        std::array<std::size_t, 4>{0,7,4,1},
        std::array<std::size_t, 4>{0,8,2,1},
        std::array<std::size_t, 4>{0,9,2,1},
        std::array<std::size_t, 4>{0,10,0,1},
    };
    
    std::array<double, NI*N_EFT*KMAX*NQ> response_coeff{};
    for (const auto& index : unity_indices)
        response_coeff[index_erc(index)] = 1.0;
    
    response_coeff[index_erc({1,2,2,1})] = 2.0;
    response_coeff[index_erc({1,6,2,0})] = 1.0/sqrt(2.0);

    return response_coeff;
}

constexpr double mass_param(double dm_mass, double nucleus_mass)
{
    const double r = dm_mass/(dm_mass + nucleus_mass);
    return 2.0*nucleus_mass*r*r;
}

constexpr std::array<double, N_EFT> dm_spin_coeff(double spin)
{
    const double c = sqrt(spin*(spin + 1.0)/3.0);
    return {1.0, 1.0, 1.0, c, -c, c, 1.0, c, c, 1.0, c};
}

void compute_nuclear_eft_expansion_coeffs(
    std::span<std::array<double, 2>>& out, const std::array<double, N_EFT*NI>& eft_coeffs, double spin, double nucleus_mass, std::span<double> form_factors)
{
    if (out.size() != form_factors.size())

    const std::array<double, N_EFT> spin_coeff = dm_spin_coeff(spin);
    std::array<double, N_EFT*NI> spun_coeffs = eft_coeffs;

    for (std::size_t i = 0; i < N_EFT; ++i)
    {
        spun_coeffs[2*i] *= spin_coeff[i];
        spun_coeffs[2*i + 1] *= spin_coeff[i];
    }

    const std::array<double, NI*N_EFT*KMAX*NQ> response_coeff
        = eft_response_coeff();
    

    /* FINISH IMPLEMENTATION */
}