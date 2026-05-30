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
#include <limits>
#include <span>
#include <vector>

#include <zest/md_span.hpp>
#include <zest/rotor.hpp>
#include <zest/sh_glq_transformer.hpp>
#include <zest/zernike_expansion.hpp>
#include <zest/zernike_glq_transformer.hpp>

#include "vector.hpp"
#include "types.hpp"
#include "zebra_angle_integrator_core.hpp"
#include "zernike_recursions.hpp"

namespace zdm::zebra
{

namespace detail
{

class ZernikeExpansionWorkspace
{
public:
    ZernikeExpansionWorkspace() = default;

    template <RadonType radon_type>
    ZernikeExpansionWorkspace(std::size_t order):
        m_data{(radon_type == RadonType::regular) ? 2 : 7, order + 4} {}

    template <RadonType radon_type>
    void expand(std::size_t order)
    {
        if constexpr (radon_type == RadonType::transverse)
            m_data.reshape(7, order + 4);
        else
            m_data.reshape(std::get<0>(m_data.extents())[0], order + 4);
    }

    [[nodiscard]] ZernikeSpan<double> geg_zernike_expansion() noexcept { return m_data[0]; };
    [[nodiscard]] ZernikeSpan<double> rotated_geg_zernike_expansion() noexcept { return m_data[1]; };
    [[nodiscard]] ZernikeSpan<double> geg_zernike_expansion_x() noexcept { return m_data[2]; };
    [[nodiscard]] ZernikeSpan<double> geg_zernike_expansion_y() noexcept { return m_data[3]; };
    [[nodiscard]] ZernikeSpan<double> geg_zernike_expansion_z() noexcept { return m_data[4]; };
    [[nodiscard]] ZernikeSpan<double> geg_zernike_expansion_r2() noexcept { return m_data[5]; };
    [[nodiscard]] ZernikeSpan<double> rotated_transverse_zernike_expansion() noexcept { return m_data[6]; };

    [[nodiscard]] ZernikeSpan<const double> geg_zernike_expansion() const noexcept { return m_data[0]; };
    [[nodiscard]] ZernikeSpan<const double> rotated_geg_zernike_expansion() const noexcept { return m_data[1]; };
    [[nodiscard]] ZernikeSpan<const double> geg_zernike_expansion_x() const noexcept { return m_data[2]; };
    [[nodiscard]] ZernikeSpan<const double> geg_zernike_expansion_y() const noexcept { return m_data[3]; };
    [[nodiscard]] ZernikeSpan<const double> geg_zernike_expansion_z() const noexcept { return m_data[4]; };
    [[nodiscard]] ZernikeSpan<const double> geg_zernike_expansion_r2() const noexcept { return m_data[5]; };
    [[nodiscard]] ZernikeSpan<const double> rotated_transverse_zernike_expansion() const noexcept { return m_data[6]; };

private:
    zest::zt::ZernikeExpansionVectorNormalGeo<double, zest::IndexingMode::zero_based> m_data;
};

} // namespace detail

template <DistType dist_type, RespType resp_type>
class AngleIntegrator {};

template <>
class AngleIntegrator<DistType::iso, RespType::iso>
{
public:
    AngleIntegrator() = default;
    explicit AngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    void integrate(
        IsotropicZernikeSpan<const double> distribution,
        std::span<const la::Vector<double, 3>> offsets, std::span<const double> shells,
        zest::DynamicMDSpan<double, 2> out);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<double> out);

private:
    IsotropicZernikeExpansion<double> m_geg_zernike_exp;
    detail::AngleIntegratorCore<DistType::iso, RespType::iso> m_integrator_core;
    std::size_t m_dist_order{};
};

template<>
class AngleIntegrator<DistType::iso, RespType::aniso>
{
public:
    AngleIntegrator() = default;
    explicit AngleIntegrator(std::size_t dist_order, std::size_t resp_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    void resize(std::size_t dist_order, std::size_t resp_order);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<double, 2> out);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<double> out);

private:
    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    IsotropicZernikeExpansion<double> m_geg_zernike_exp;
    detail::AngleIntegratorCore<DistType::iso, RespType::aniso> m_integrator_core;
    std::size_t m_dist_order{};
    std::size_t m_resp_order{};
};

/**
    @brief Angle integrated Radon transforms using the Zernike based Radon transform.
*/
template <>
class AngleIntegrator<DistType::aniso, RespType::iso>
{
public:
    AngleIntegrator() = default;
    explicit AngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated Radon transform of a disitribution on an offset unit ball.

        @param distribution Zernike expansion of distribution
        @param offsets offsets of the distribution
        @param shells distances of integration planes from the origin
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> shells, zest::DynamicMDSpan<double, 2> out);

    /**
        @brief Angle integrated Radon transform of a disitribution on an offset unit ball.

        @param distribution Zernike expansion of the distribution
        @param offset offset of the distribution
        @param shells distances of integration planes from the origin
        @param out output values as 1D array of length `shells.size()`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<double> out);

private:
    void integrate(
        const la::Vector<double, 3>& offset, std::span<const double> shells,
        std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    ZernikeExpansion<double> m_geg_zernike_exp;
    ZernikeExpansion<double> m_rotated_geg_zernike_exp;
    detail::ZernikeExpansionWorkspace m_zernike_expansions;
    detail::AngleIntegratorCore<DistType::aniso, RespType::iso> m_integrator_core;
    std::size_t m_dist_order{};
};

/**
    @brief Angle integrated Radon transforms with anisotropic response function using the
    Zernike based Radon transform.
*/
template <>
class AngleIntegrator<DistType::aniso, RespType::aniso>
{
public:
    AngleIntegrator() = default;
    AngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    [[nodiscard]] std::size_t
    truncation_order() const noexcept { return m_trunc_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution
        @param response spherical harmonic expansions of response on `shells`
        @param offsets offsets of the distribution
        @param rotation_angles z-axis rotation angles between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`
        @param trunc_order maximum truncation order for internal expansions

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        Assuming the Zernike expansion of the distribution has order `L` and the
        spherical harmonic expansions of the response have order `K`, the algorithm will
        internally employ expansions of orders up to `K + L` to avoid aliasing when
        taking products of the distribution and response. However, in practice, the
        aliasing could be insignificant. Therefore, the parameter `trunc_order` is
        provided, which can cap the order of the internal expansions.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<double, 2> out,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution
        @param response spherical harmonic expansions of response on `shells`
        @param offset offset of the distribution
        @param rotation_angle z-axis rotation angle between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param out output values as 1D array of size `shells.size()`
        @param trunc_order maximum truncation order for internal expansions

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        Assuming the Zernike expansion of the distribution has order `L` and the
        spherical harmonic expansions of the response have order `K`, the algorithm will
        internally employ expansions of orders up to `K + L` to avoid aliasing when
        taking products of the distribution and response. However, in practice, the
        aliasing could be insignificant. Therefore, the parameter `trunc_order` is
        provided, which can cap the order of the internal expansions.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<double> out,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::size_t geg_order,
        std::size_t top_order, std::span<double> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    ZernikeExpansion<double> m_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_grids;
    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    detail::AngleIntegratorCore<DistType::aniso, RespType::aniso> m_integrator_core;
    std::size_t m_dist_order{};
    std::size_t m_resp_order{};
    std::size_t m_trunc_order{};
};

template <DistType dist_type, RespType resp_type>
class TransverseAngleIntegrator {};

template<>
class TransverseAngleIntegrator<DistType::iso, RespType::iso>
{
public:
    TransverseAngleIntegrator() = default;
    explicit TransverseAngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> shells, zest::DynamicMDSpan<std::array<double, 2>, 2> out);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<std::array<double, 2>> out);

private:
    IsotropicZernikeExpansion<double, 3> m_transverse_geg_zernike_exp_components;
    detail::IsotropicZernikeTransverseRadonHelper m_transverse_radon_helper;
    detail::AngleIntegratorCore<DistType::iso, RespType::iso> m_integrator_core;
    std::size_t m_dist_order{};
};

template<>
class TransverseAngleIntegrator<DistType::iso, RespType::aniso>
{
public:
    TransverseAngleIntegrator() = default;
    TransverseAngleIntegrator(std::size_t dist_order, std::size_t resp_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    void resize(std::size_t dist_order, std::size_t resp_order);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<std::array<double, 2>, 2> out);

    void integrate(
        IsotropicZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<std::array<double, 2>> out);

private:
    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    IsotropicZernikeExpansion<double, 3> m_transverse_geg_zernike_exp_components;
    detail::IsotropicZernikeTransverseRadonHelper m_transverse_radon_helper;
    detail::AngleIntegratorCore<DistType::iso, RespType::aniso> m_integrator_core;
    std::size_t m_dist_order{};
    std::size_t m_resp_order{};
};

/**
    @brief Angle integrated regular and transverse Radon transforms and using the Zernike
    based Radon transform.
*/
template <>
class TransverseAngleIntegrator<DistType::aniso, RespType::iso>
{
public:
    TransverseAngleIntegrator() = default;
    explicit TransverseAngleIntegrator(std::size_t dist_order);

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    void resize(std::size_t dist_order);

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a
        velocity disitribution on an offset unit ball.

        @param distribution Zernike expansion of the distribution
        @param offsets offsets of the distribution
        @param shells distances of integration planes from the origin
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> shells, zest::DynamicMDSpan<std::array<double, 2>, 2> out);

    /**
        @brief Angle integrated transverse and nontransverse Radon transform of a
        velocity disitribution on an offset unit ball.

        @param distribution Zernike expansion of the distribution
        @param offset offset of the distribution
        @param shells distances of integration planes from the origin
        @param out output values as 1D array of size `shells.size()`

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, const la::Vector<double, 3>& offset,
        std::span<const double> shells, std::span<std::array<double, 2>> out);

private:
    void integrate(
        const la::Vector<double, 3>& offset, std::span<const double> shells,
        std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    zest::Rotor m_rotor;
    ZernikeExpansion<double> m_geg_zernike_exp;
    ZernikeExpansion<double> m_geg_zernike_exp_x;
    ZernikeExpansion<double> m_geg_zernike_exp_y;
    ZernikeExpansion<double> m_geg_zernike_exp_z;
    ZernikeExpansion<double> m_geg_zernike_exp_r2;
    ZernikeExpansion<double> m_rotated_geg_zernike_exp;
    ZernikeExpansion<double> m_rotated_trans_geg_zernike_exp;
    detail::ZernikeCoordinateMultiplier m_multiplier;
    detail::AngleIntegratorCore<DistType::aniso, RespType::iso> m_integrator_core;
    std::size_t m_dist_order{};
};

/**
    @brief Angle integrated regular and transverse Radon transforms with anisotropic
    response function using the Zernike based Radon transform.
*/
template <>
class TransverseAngleIntegrator<DistType::aniso, RespType::aniso>
{
public:
    TransverseAngleIntegrator() = default;
    TransverseAngleIntegrator(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    [[nodiscard]] std::size_t
    distribution_order() const noexcept { return m_dist_order; }

    [[nodiscard]] std::size_t
    response_order() const noexcept { return m_resp_order; }

    [[nodiscard]] std::size_t
    truncation_order() const noexcept { return m_trunc_order; }

    void resize(
        std::size_t dist_order, std::size_t resp_order,
        std::size_t trunc_order);

    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution
        @param response spherical harmonic expansions of response on `shells`
        @param offsets offsets of the distribution
        @param rotation_angles z-axis rotation angles between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param out output values as 2D array of shape `{offsets.size(), shells.size()}`
        @param trunc_order maximum truncation order for internal expansions

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        Assuming the Zernike expansion of the distribution has order `L` and the
        spherical harmonic expansions of the response have order `K`, the algorithm will
        internally employ expansions of orders up to `K + L` to avoid aliasing when
        taking products of the distribution and response. However, in practice, the
        aliasing could be insignificant. Therefore, the parameter `trunc_order` is
        provided, which can cap the order of the internal expansions.

        @note The offsets and rotation angles come in pairs. Therefore `offsets` and
        `rotation_angles` must have the same size.

        @note `distribution` and `offsets` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        std::span<const la::Vector<double, 3>> offsets,
        std::span<const double> rotation_angles, std::span<const double> shells,
        zest::DynamicMDSpan<std::array<double, 2>, 2> out,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

    /**
        @brief Angle integrated Radon transform of a disitribution on a offset unit ball,
        combined with an angle-dependent response.

        @param distribution Zernike expansion of the distribution
        @param response spherical harmonic expansions of response on `shells`
        @param offset offset of the distribution
        @param rotation_angle z-axis rotation angle between distribution and response coordinates
        @param shells distances of integration planes from the origin
        @param out output values as 1D array of size `shells.size()`
        @param trunc_order maximum truncation order for internal expansions

        The 3D Radon transform is defined as an integral over a plane in the 3D space.
        Any given plane is uniquely determined by its unit normal vector, and its
        distance to the origin, defined by the plane's nearest point to the origin.
        The set of unit vectors at a given distance define a spherical shell. Hence
        the angle-integrated Radon transform is parametrized by the distances, given
        in the `shells` parameter.

        The Radon transform has well-defined transformation properties under affine
        transforms. In the angle-integrated Radon transform, the only relevant part
        of the affine transform is the offset. Thus the `offsets` parameter contains
        an arbitrary collection of offset vectors for the distribution.

        The response is an arbitrary function of the unit normal vector and shell
        radius. It multiplies the Radon transform before the integration over unit
        normals.

        The response can be defined in a coordinate system, which differs from the
        response coordinate system by an arbitrary rotation about the z-axis. The
        angles of these rotations are given in the `rotation_angles` parameter. The
        rotation angles specifically is the counterclockwise angle from the x-axis of the
        response coordinate system to the x-axis of the distribution coordinate system.

        Assuming the Zernike expansion of the distribution has order `L` and the
        spherical harmonic expansions of the response have order `K`, the algorithm will
        internally employ expansions of orders up to `K + L` to avoid aliasing when
        taking products of the distribution and response. However, in practice, the
        aliasing could be insignificant. Therefore, the parameter `trunc_order` is
        provided, which can cap the order of the internal expansions.

        @note `distribution` and `offset` are defined in the same coordinates.
    */
    void integrate(
        ZernikeSpan<const double> distribution, SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<std::array<double, 2>> out,
        std::size_t trunc_order = std::numeric_limits<std::size_t>::max());

private:
    void integrate(
        SHVectorSpan<const double> response,
        const la::Vector<double, 3>& offset, double rotation_angle,
        std::span<const double> shells, std::span<std::array<double, 2>> out);

    zest::WignerdPiHalfCollection m_wigner_d_pi2;
    ZernikeExpansion<double> m_geg_zernike_exp;
    ZernikeExpansion<double> m_geg_zernike_exp_x;
    ZernikeExpansion<double> m_geg_zernike_exp_y;
    ZernikeExpansion<double> m_geg_zernike_exp_z;
    ZernikeExpansion<double> m_geg_zernike_exp_r2;
    std::vector<double> m_rotated_geg_zernike_exp;
    ZernikeExpansion<double> m_rotated_trans_geg_zernike_exp;
    std::vector<double> m_rotated_geg_zernike_grids;
    std::vector<double> m_rotated_trans_geg_zernike_grids;
    detail::ZernikeCoordinateMultiplier m_multiplier;
    zest::Rotor m_rotor;
    zest::st::GLQTransformerGeo<> m_glq_transformer;
    detail::AngleIntegratorCore<DistType::aniso, RespType::aniso> m_integrator_core;
    std::size_t m_dist_order{};
    std::size_t m_resp_order{};
    std::size_t m_trunc_order{};
};

} // namespace zdm::zebra


