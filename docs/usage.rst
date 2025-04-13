Integrating a Radon transform
=============================

This section goes through the core parts of the library and its usage via an example of evaluating the angle-integrated Radon transform of a distribution with a response function.

The angle integration is implemented in four core classes: :cpp:type:`zdm::zebra::IsotropicAngleIntegrator`, :cpp:type:`zdm::zebra::AnisotropicAngleIntegrator`, :cpp:type:`zdm::zebra::IsotropicTransverseAngleIntegrator`, :cpp:type:`zdm::zebra::AnisotropicTransverseAngleIntegrator`. The classes with ``Transverse`` compute the angle-integrated transverse Radon transform in addition to the nontransverse case, whereas the others only compute the nontransverse case. The classes marked with ``Anisotropic`` are used when we have an anisotropic response function, whereas the ``Isotropic`` classes are for the special case where the response function is istropic (or, alternatively, when we have no response function).

We will mainly consider :cpp:type:`zdm::zebra::AnisotropicAngleIntegrator` here. The isotropic integrators are simpler, because they don't need to deal with the presence of a response function, and the transverse integrators in turn essentially only differ by the fact that they return two numbers where the nontransverse integrators return one.

The first order of business is to initialize our integrator

.. code:: cpp

    #include "zebradm/zebra_angle_integrator.hpp"

    int main()
    {
        constexpr std::size_t dist_order = 30;
        constexpr std::size_t resp_order = 60;

        // ...

        zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);

        // ...
    }

All the integrators are located in the header ``zebradm/zebra_angle_integrator.hpp``. The parameters ``dist_order`` and ``resp_order`` are the orders of the Zernike and spherical harmonic expansions of the distribution and response functions, respectively. One can also default initialize the integrator

.. math::

    zdm::zebra::AnisotropicAngleIntegrator integrator{};

Initializing with the order parameter preallocates some buffers. However, when the integrator is used, it will also read these paramters off the expansions it is given, and adjust its buffers accordingly, so whether to use default initialization or not is a matter of preference.

In order to use the integrator, we will need to prepare the data needed to compute the Radon transform. First and foremost, we need a Zernike expansion representing our distribution function. Typically, the Zernike expansion is computed from some mathematical expression for the distribution, so we will define one

.. code:: cpp

    constexpr std::array<double, 3> dispersion = {0.5, 0.6, 0.7};
    auto dist_func = [&](double r, double lon, double colat)
    {
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
        };

        return std::exp(-(x[0]*x[0]/dispersion[0]) + x[1]*x[1]/dispersion[1] + x[2]*x[2]/dispersion[2]);
    };

This is a C++ lambda function describing an anisotropic Gaussian distribution. The function takes three doubles denoting the three spherical coordinates. The distribution function must have either this signature, or an alternative signature which takes a single ``std::array<double, 3>``, denoting the Cartesian three-vector ``x``. Defining the distribution function as a lambda, because additional parameters can be taken as captures, as is the case with ``dispersion`` here.

The business of Zernike and spherical harmonic transforms and expansions is handled by the library `zest <https://github.com/sebsassi/zest>`. We can use zest's :cpp:type:`zest::zt::ZernikeTransformer` to accomplish this. As a more general purpose library, zest supports multiple conventions for normalization and the Condon--Shortley phase. In ZebraDM the conventions are chosen to be such that the spherical harmonics are :math:`4\pi`-normalized and defined without the Condon--Shortley phase, and the radial Zernike polynomials are fully normalized. Multiple aliases of the basic types are defined by zest for different combinations of conventions, and so the correct transformer for Zernike expansions compatible with ZebraDM is :cpp:type:`zest::zt::ZernikeTransformerNormalGeo`. We can use this to easily get the Zernike expansion of our distribution

.. code:: cpp

    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo(dist_order).transform(dist_func);

The next problem is to define our response function. For purposes of this demonstration, we use an arbitrary function

.. code:: cpp

    constexpr std::array<double, 3> a = {0.5, 0.5, 0.5};
    auto resp_func = [&](double min_speed, double lon, double colat)
    {
        const std::array<double, 3> dir = {
            std::sin(colat)*std::cos(lon), std::sin(colat)*std::sin(lon), std::cos(colat)
        };

        return std::exp(-min_speed*(zdm::linalg(dir, a)));
    };

The argument ``min_speed`` here is same as the parameter :math:`w` used in the theoretical description, which in dark matter direct detection literature is often denoted :math:`v_\text{min}`. In nuclear scattering of dark matter this is the minimum speed needed from dark matter to give the nucleus recoil momentum equal to the momentum transfer.


