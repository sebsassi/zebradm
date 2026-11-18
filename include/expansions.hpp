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

#include <cstddef>
#include <concepts>

#include <zest/sh_conventions.hpp>
#include <zest/sh_expansion.hpp>
#include <zest/zernike_expansion.hpp>

#include "shapes.hpp"

namespace zdm
{

/**
    @brief Alias for spherical harmonic expansion with conventions used by
    zebradm.

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
using SHExpansion = zest::st::SHExpansion<
    ElementType, zest::Indexing::zero_based, zest::st::Geo, inner_extents...>;

/**
    @brief Alias for a non-owning view to a spherical harmonic expansion with
    conventions used by zebradm.

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
using SHSpan = zest::st::SHSpan<
    ElementType, zest::Indexing::zero_based, zest::st::Geo, inner_extents...>;

/**
    @brief Alias for Zernike expansion with conventions used by zebradm.

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
using ZernikeExpansion = zest::zt::ZernikeExpansion<
    double, zest::Indexing::zero_based, zest::zt::NormedGeo, inner_extents...>;

/**
    @brief Alias for a non-owning view to a Zernike expansion with conventions
    used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
    using ZernikeSpan = zest::zt::ZernikeSpan<
    ElementType, zest::Indexing::zero_based, zest::zt::NormedGeo, inner_extents...>;

/**
    @brief Alias for isotropic Zernike expansion with conventions used by
    zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansion = zest::zt::IsotropicZernikeExpansion<
    ElementType, zest::zt::NormedGeo, inner_extents...>;

/**
    @brief Alias for a non-owning view to a Zernike expansion with conventions
    used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
    @tparam inner_extents Extents of an inner multi-dimensional array structure.
*/
template <std::floating_point ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpan = zest::zt::IsotropicZernikeSpan<
    ElementType, zest::zt::NormedGeo, inner_extents...>;

/**
    @brief Alias for an array of spherical harmonic expansions with conventions
    used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
*/
template <std::floating_point ElementType>
using SHExpansionVector = zest::st::SHExpansionVector<
    double, zest::Indexing::zero_based, zest::st::Geo>;

/**
    @brief Alias for a non-owning view of an array of spherical harmonic
    expansions with conventions used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
*/
template <std::floating_point ElementType>
using SHVectorSpan = zest::st::SHVectorSpan<
    ElementType, zest::Indexing::zero_based, zest::st::Geo>;

/**
    @brief Alias for a non-owning view to an array of Zernike expansions with conventions
    used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
*/
template <std::floating_point ElementType>
using IsotropicZernikeVectorSpan = zest::zt::IsotropicZernikeVectorSpan<
    ElementType, zest::zt::NormedGeo>;

/**
    @brief Alias for a non-owning view to an array of Zernike expansions with conventions
    used by zebradm.

    @tparam ElementType

    @tparam ElementType Type of expansion coefficients.
*/
template <std::floating_point ElementType>
using IsotropicZernikeExpansionVector = zest::zt::IsotropicZernikeExpansionVector<
    ElementType, zest::zt::NormedGeo>;

template <std::floating_point ElementType, std::size_t... inner_extents>
using LegendreShellExpansion
    = zest::ShapedArray<ElementType, LegendreShellShape<inner_extents...>>;

template <std::floating_point ElementType, std::size_t... inner_extents>
using LegendreShellSpan
    = zest::ShapedSpan<ElementType, LegendreShellShape<inner_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using LegendreShellExpansionTensor
    = zest::ShapedArray<ElementType, LegendreShellTensorShape<outer_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using LegendreShellTensorSpan
    = zest::ShapedSpan<ElementType, LegendreShellTensorShape<outer_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using IsotropicLegendreShellExpansion
    = zest::ShapedArray<ElementType, IsotropicLegendreShellShape<outer_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using IsotropicLegendreShellSpan
    = zest::ShapedSpan<ElementType, IsotropicLegendreShellShape<outer_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using IsotropicLegendreShellExpansionTensor
    = zest::ShapedArray<ElementType, IsotropicLegendreShellTensorShape<outer_extents...>>;

template <std::floating_point ElementType, std::size_t... outer_extents>
using IsotropicLegendreShellTensorSpan
    = zest::ShapedSpan<ElementType, IsotropicLegendreShellTensorShape<outer_extents...>>;

template <std::floating_point ElementType>
using LegendreShellExpansionVector
    = LegendreShellExpansionTensor<ElementType, std::dynamic_extent>;

template <std::floating_point ElementType>
using LegendreShellVectorSpan
    = LegendreShellTensorSpan<ElementType, std::dynamic_extent>;

template <std::floating_point ElementType>
using IsotropicLegendreShellGrid
    = zest::ShapedArray<ElementType, zest::TensorShape<std::dynamic_extent>>;

template <std::floating_point ElementType>
using IsotropicLegendreShellGridSpan
    = zest::ShapedSpan<ElementType, zest::TensorShape<std::dynamic_extent>>;


} // namespace zdm
