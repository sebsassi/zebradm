Usage -- Integrating a Radon transform
======================================

This section goes through the core parts of the library and its usage via an example of evaluating
the angle-integrated Radon transform of a distribution with a response function.

The angle integration is implemented in four core classes:

* :cpp:type:`zdm::zebra::IsotropicAngleIntegrator`
* :cpp:type:`zdm::zebra::AnisotropicAngleIntegrator`
* :cpp:type:`zdm::zebra::IsotropicTransverseAngleIntegrator`
* :cpp:type:`zdm::zebra::AnisotropicTransverseAngleIntegrator`

The classes with ``Transverse`` compute the angle-integrated transverse Radon transform in addition
to the nontransverse case, whereas the others only compute the nontransverse case. The classes
marked with ``Anisotropic`` are used when we have an anisotropic response function, whereas the
``Isotropic`` classes are for the special case where the response function is istropic (or,
alternatively, when we have no response function).

We will mainly consider :cpp:type:`zdm::zebra::AnisotropicAngleIntegrator` here. The isotropic
integrators are simpler, because they don't need to deal with the presence of a response function,
and the transverse integrators in turn essentially only differ by the fact that they return two
numbers where the nontransverse integrators return one.

The first order of business is to initialize our integrator

.. code:: cpp

    #include <zebradm/zebra_angle_integrator.hpp>

    int main()
    {
        constexpr std::size_t dist_order = 30;
        constexpr std::size_t resp_order = 60;

        // ...

        zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);

        // ...
    }

All the integrators are located in the header ``zebradm/zebra_angle_integrator.hpp``. The
parameters ``dist_order`` and ``resp_order`` are the orders of the Zernike and spherical harmonic
expansions of the distribution and response functions, respectively. One can also default
initialize the integrator

.. code:: cpp

    zdm::zebra::AnisotropicAngleIntegrator integrator{};

Initializing with the order parameter preallocates some buffers. However, when the integrator is
used, it will also read these paramters off the expansions it is given, and adjust its buffers
accordingly, so whether to use default initialization or not is a matter of preference.

In order to use the integrator, we will need to prepare the data needed to compute the Radon
transform. First and foremost, we need a Zernike expansion representing our distribution function.
Typically, the Zernike expansion is computed from some mathematical expression for the
distribution, so we will define one

.. code:: cpp

    #include <cmath>
    #include <array>

    constexpr std::array<double, 3> dispersion = {0.5, 0.6, 0.7};
    auto dist_func = [&](double lon, double colat, double r)
    {
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
        };

        return std::exp(-(x[0]*x[0]/dispersion[0]) + x[1]*x[1]/dispersion[1] + x[2]*x[2]/dispersion[2]);
    };

This is a C++ lambda function describing an anisotropic Gaussian distribution. The function takes
three doubles denoting the three spherical coordinates. The distribution function must have either
this signature, or an alternative signature which takes a single ``std::array<double, 3>``,
denoting the Cartesian three-vector ``x``. Defining the distribution function as a lambda, because
additional parameters can be taken as captures, as is the case with ``dispersion`` here.

The business of Zernike and spherical harmonic transforms and expansions is handled by the library
`zest <https://github.com/sebsassi/zest>`_. We can use zest's :cpp:type:`zest::zt::ZernikeTransformer`
to accomplish this. As a more general purpose library, zest supports multiple conventions for
normalization and the Condon--Shortley phase. In ZebraDM the conventions are chosen to be such that
the spherical harmonics are :math:`4\pi`-normalized and defined without the Condon--Shortley phase,
and the radial Zernike polynomials are fully normalized. Multiple aliases of the basic types are
defined by zest for different combinations of conventions, and so the correct transformer for
Zernike expansions compatible with ZebraDM is :cpp:type:`zest::zt::ZernikeTransformerNormalGeo`.
We can use this to easily get the Zernike expansion of our distribution

.. code:: cpp

    #include <zest/zernike_glq_transformer.hpp>
    
    constexpr double radius = 2.0;
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo{}.transform(dist_func, radius, dist_order);

The Zernike functions are defined on the unit ball, but we can obviously scale any ball to a unit
ball. The ``radius`` parameter here does exactly that. It is the radius of the ball on which our
function is defined, so that :cpp:type:`zest::zt::ZernikeTransformer` can do the scaling for you.

The next problem is to define our response function. For purposes of this demonstration, we use an
arbitrary function

.. code:: cpp

    constexpr std::array<double, 3> a = {0.5, 0.5, 0.5};
    auto resp_func = [&](double shell, double lon, double colat)
    {
        const std::array<double, 3> dir = {
            std::sin(colat)*std::cos(lon), std::sin(colat)*std::sin(lon), std::cos(colat)
        };

        return std::exp(-min_speed*(zdm::linalg(dir, a)));
    };

The argument ``shell`` here is same as the shell parameter :math:`w` (see the section on
theoretical background), which in dark matter direct detection literature is often denoted
:math:`v_\text{min}`. In nuclear scattering of dark matter this is the minimum speed needed from
dark matter to give the nucleus recoil momentum equal to the momentum transfer.

The angle-integrated Radon transform in this library is defined on a collection of shell
parameters. We therefore need to decide upon the collection of shell parameters. As discussed in
the theoretical background section, the geometry of the situation means that if our distribution
has offset :math:`\vec{x}_0`, then the angle-integrated Radon transform goes to zero for
:math:`w > 1 + x_0`. Therefore, to determine an appropriate maximum value for the shell parameter,
we will need to determine our offsets. In a real problem the offsets would come from somewhere. For
example, in the context of dark matter direct detection they are the velocities of the laboratory
relative to the dark matter distribution. For purposes of this example, we will generate a random
list of vectors of some length

.. code:: cpp

    #include <random>
    #include <vector>

    std::vector<std::array<double, 3>> generate_offsets(std::size_t count, double offset_len)
    {
        std::mt19937 gen;
        std::uniform_real_distribution rng_dist{0.0, 1.0};

        std::vector<std::array<double, 3>> offsets(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const double ct = 2.0*rng_dist(gen) - 1.0;
            const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
            const double az = 2.0*std::numbers::pi*rng_dist(gen);
            offsets[i] = {offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct};
        }
        
        return offsets;
    }

Alongside this, we can create a similar function that generates a vector of shell parameters

.. code:: cpp

    std::vector<double> generate_shells(std::size_t count, double offset_len)
    {
        const double max_shell = 1.0 + offset_len;

        std::vector<double> shells(count);
        for (std::size_t i = 0; i < count; ++i)
            shells[i] = max_shell*double(i)/double(count - 1);

        return shells;
    }

Then we can generate the offsets and shells

.. code:: cpp

    constexpr double offset_len = 0.5;
    constexpr double offset_count = 10;
    constexpr double shell_count = 50;

    std::vector<std::array<double, 3>> offsets = generate_offsets(offset_count, offset_len);
    std::vector<double> shells = generate_shells(shell_count, offset_len);

Now that we actually have the shells, we can compute the spherical harmonic transforms of the
shells on the response functions. For this purpose, the header ``zebradm/zebra_util.hpp`` provides
the container :cpp:type:`zdm::SHExpansionVector` for storing a collection of spherical harmonic
expansions in a single buffer, as well as the class :cpp:type:`zdm::zebra::ResponseTransformer`
for computing the spherical harmonic expansions.

.. code:: cpp

    zdm::SHExpansionVector response 
        = zdm::zebra::ResponseTransformer{}.transform(resp_func, shells, resp_order);

At this point we are almost ready to use the integrator. We still need two things, however. First
is a vector of rotation angles for each offset, because not only can the distribution be defined in
coordinates with an arbitrary offset, but it can also have a rotation relative to the coordinates
in which the response is defined.

In principle, the distribution and response functions could be defined in coordinate systems which
differ from each other by an arbitrary 3D rotation. However, arbitrary 3D rotations of spherical
harmonic expansions are expensive, so the transformer has been limited to doing rotations about the
z-axis per offset. With that said, nothing stops you from applying arbitrary global rotations on
the expansions of the distribution and response before handing them off to the integrator. You can
rotate Zernike and spherical harmonic transforms by arbitrary Euler angles with the
:cpp:type:`zest::Rotor` class

.. code:: cpp

    #include <numbers>

    #include <zest/rotor.hpp>

    zest::Rotor rotor(resp_order);
    constexpr std::array<double, 3> euler_angles = {
        std::numbers::pi/2, std::numbers::pi/3, std::numbers::pi/4
    };

    for (std::size_t i = 0; i < response.extent(); ++i)
        rotor.rotate(response[i], euler_angles, zest::RotatioType::coordinate);

The last argument here tells the rotor whether we are rotating the coordinate system (active), or
the object (passive). You can read more about this in the zest documentation.

With that said, here we can just create a nice full rotation

.. code:: cpp

    std::vector<double> generate_rotation_angles(std::size_t offset_count)
    {
        std::vector<double> rotation_angles(offset_count);
        for (std::size_t i = 0; i < offset_count; ++i)
            rotation_angles[i] = 2.0*std::numbers::pi*double(i)/double(offset_count - 1);
    }

and then generate the rotation angles

.. code:: cpp

    std::vector<double> rotation_angles = generate_rotation_angles(offset_count);

Now, the last remaining thing we need is a buffer to put the results in

.. code:: cpp

    #include <zest/md_array.hpp>

    zest::MDArray<double, 2> out({offset_count, shell_count});

If we were dealing with one of the ``Transverse`` integrators, then then we would have to use
:cpp:type:`std::array\<double, 2>` as the element type of ``out`` instead to store the
nontransverse--transverse pair.

With this, we finally have everything in place to integrate the angle-integrated Radon transform

.. code:: cpp

    integrator.integrate(distribution, response, offsets, rotation_angles, shells, out);

Now, this is almost it. However, there is one point which need to be accounted for. Earlier we set
the parameter ``radius = 2.0`` indicating to the Zernike transformer that our distribution is
defined in a ball of radius two. However, the Radon transform is always evaluated on the unit ball.
This means that if we defined the unit ball coordinates :math:`\vec{x} = \vec{r}/R`, where :math:`R`
is our radius, then

.. math::

    \mathcal{R}[f](w,\hat{n})
        = \int\delta(\vec{r}\cdot\hat{n} - w)f(\vec{r})\,d^3r = R^2\int\delta(\vec{x}\cdot\hat{n} - w/R)f(R\vec{x})\,d^3x.

That is, in practice, not only do we need to divide our original shell parameters by the radius
(which we didn't do here because we just generated the scaled parameters directly), but we also
have to multiply our result by the radius squared

.. code:: cpp

    for (auto& element : out.flatten())
        element *= radius*radius;

If we were also evaluating the transverse Radon transform, we would likewise have to multiply it by
the fourth power of the radius.

And this is it. We have successfully computed the angle-integrated Radon transform of of our
distribution, combined with an anisotropic response function, for a set of shells and offset--angle
pairs. In summary, here is the full source code of our program

.. code:: cpp

    #include <array>
    #include <vector>
    #include <cmath>
    #include <numbers>
    #include <random>
    #include <cstdio>

    #include <zest/zernike_glq_transformer.hpp>
    #include <zest/md_array.hpp>
    #include <zest/rotor.hpp>

    #include <zebradm/zebra_angle_integrator.hpp>

    std::vector<std::array<double, 3>> generate_offsets(std::size_t count, double offset_len)
    {
        std::mt19937 gen;
        std::uniform_real_distribution rng_dist{0.0, 1.0};

        std::vector<std::array<double, 3>> offsets(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const double ct = 2.0*rng_dist(gen) - 1.0;
            const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
            const double az = 2.0*std::numbers::pi*rng_dist(gen);
            offsets[i] = {offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct};
        }

        return offsets;
    }

    std::vector<double> generate_rotation_angles(std::size_t offset_count)
    {
        std::vector<double> rotation_angles(offset_count);
        for (std::size_t i = 0; i < offset_count; ++i)
            rotation_angles[i] = 2.0*std::numbers::pi*double(i)/double(offset_count - 1);
    }


    std::vector<double> generate_shells(std::size_t count, double offset_len)
    {
        const double max_shell = 1.0 + offset_len;

        std::vector<double> shells(count);
        for (std::size_t i = 0; i < count; ++i)
            shells[i] = max_shell*double(i)/double(count - 1);

        return shells;
    }

    int main()
    {
        constexpr std::array<double, 3> dispersion = {0.5, 0.6, 0.7};
        auto dist_func = [&](double lon, double colat, double r)
        {
            const std::array<double, 3> x = {
                r*std::sin(colat)*std::cos(lon), r*std::sin(colat)*std::sin(lon), r*std::cos(colat)
            };

            return std::exp(-(x[0]*x[0]/dispersion[0]) + x[1]*x[1]/dispersion[1] + x[2]*x[2]/dispersion[2]);
        };

        constexpr std::array<double, 3> a = {0.5, 0.5, 0.5};
        auto resp_func = [&](double shell, double lon, double colat)
        {
            const std::array<double, 3> dir = {
                std::sin(colat)*std::cos(lon), std::sin(colat)*std::sin(lon), std::cos(colat)
            };

            return std::exp(-min_speed*(zdm::linalg(dir, a)));
        };

        constexpr double offset_len = 0.5;
        constexpr double offset_count = 10;
        constexpr double shell_count = 50;

        std::vector<std::array<double, 3>> offsets = generate_offsets(offset_count, offset_len);
        std::vector<double> rotation_angles = generate_rotation_angles(offset_count);
        std::vector<double> shells = generate_shells(shell_count, offset_len);

        constexpr double radius = 2.0;
        constexpr std::size_t dist_order = 30;
        zest::zt::RealZernikeExpansionNormalGeo distribution
            = zest::zt::ZernikeTransformerNormalGeo{}.transform(dist_func, radius, dist_order);

        constexpr std::size_t resp_order = 60;
        zdm::SHExpansionVector response 
            = zdm::zebra::ResponseTransformer{}.transform(resp_func, shells, resp_order);

        constexpr std::array<double, 3> euler_angles = {
            std::numbers::pi/2, std::numbers::pi/3, std::numbers::pi/4
        };

        zest::Rotor rotor(resp_order);
        for (std::size_t i = 0; i < response.extent(); ++i)
            rotor.rotate(response[i], euler_angles, zest::RotatioType::coordinate);

        zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);

        zest::MDArray<double, 2> out({offset_count, shell_count});
        integrator.integrate(distribution, response, offsets, rotation_angles, shells, out);

        for (auto& element : out.flatten())
            element *= radius*radius;

        for (std::size_t i = 0; i < out.extent(0); ++i)
        {
            for (std::size_t j = 0; j < out.extent(0); ++j)
                std::printf("%.7e", out[i,j]);
            std::printf("\n");
        }
    }

