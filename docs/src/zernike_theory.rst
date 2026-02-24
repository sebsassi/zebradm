Spherical harmonics and 3D Zernike functions
============================================

The foundation of this library is the fact that large classes of functions can be expressed as
linear combinations of functions :math:`\{\psi_i(\vec{x})\}_{i\in I}` which form an orthogonal
basis, such that

.. math::

    \int\psi_i(\vec{x})\psi_j(\vec{x})\,d^nx = N_i\delta_{ij}.

In particular, this library relies on two useful basis expansions. The first is the set of
real spherical harmonics :math:`Y_{lm}(\theta,\varphi)`, which form an orthogonal basis on the
sphere :math:`S^2`, such that

.. math::

    \int_{S^2} Y_{lm}(\theta,\varphi)Y_{l'm'}(\theta,\varphi)\, d\Omega = N_{lm}\delta_{ll'}\delta_{mm'}.

The coefficient :math:`N_{lm}` is a normalization factor, which depends on the normalization
convention used. The choice of normalization convention is arbitrary, and different applications
use different conventions. This library, for no particular reason, adopts the so-called :math:`4\pi`
or "geodesy" convention, where :math:`N_{lm} = 4\pi`.

The spherical harmonics find common applications in problems with spherical symmetric from quantum
mechanics to acoustics and geodesy. The spherical harmonics are most commonly presented in terms of
the their complex form; :math:`Y_l^m(\theta,\varphi)`, which in the :math:`4\pi` normalization can
be explicitly written as

.. math::

    Y_l^m(\theta,\varphi) = \sqrt{(2l + 1)\frac{(l - m)!}{(l + m)!}}P_l^m(\cos\theta)e^{im\varphi}.

However, since this library deals solely with real functions, there are advantages to using the
real form of the spherical harmonics, which are related to the complex spherical harmonics via

.. math::

    Y_{lm}(\theta,\varphi) =
    \begin{cases}
        \sqrt{2}\text{Im}(Y_l^{|m|}(\theta,\varphi)),   & m < 0,
        Y_l^0(\theta,\varphi),                          & m = 0,
        \sqrt{2}\text{Re}(Y_l^m)(\theta,\varphi),       & m > 0.
    \end{cases}

Throughout this documentation, we use the common convention, where a raised :math:`m`-index
signifies complex spherical harmonics, while a lowered :math:`m`-index signifies real spherical
harmonics.

The other basis used by this library, which builds on the spherical harmonics, is the basis of real
3D Zernike functions :math:`Z^\alpha_{nlm}(\vec{x})`. This set of functions forms an orthogonal
basis on the unit ball :math:`B = \{\vec{x}\in\mathbb{R}^3 : \|\vec{x}\| \leq 1\}`, such that

.. math::

    \int_B Z^\alpha_{nlm}(\vec{x})Z_{n'l'm'}(\vec{x})\,d^3x = N_{nlm}\delta_{nn'}\delta_{ll'}\delta_{mm'}.

The radially unnormalized Zernike functions may be written as

.. math::

    Z^\alpha_{nlm}(\vec{x}) = R^\alpha_{nl}(r)Y_{lm}(\theta,\varphi),

where :math:`R_{nl}(r)` are the unnormalized radial Zernike polynomials, which can in turn be
written in terms of the Jacobi polynomials :math:`P^{(\alpha,\beta)}_k(x)` as

.. math::

    R^\alpha_{nl}(r) = r^l(1 - r^2)^\alpha P^{(\alpha,l + 1/2)}_{(n - l)/2}(2r^2 - 1).

The radial Zernike polynomials have the orthogonality relation

.. math::

    \int_0^1 R^\alpha_{nl}(r)R^\alpha_{n'l}\frac{r^2\,dr}{(1 - r^2)^\alpha} = N^\alpha_{nl}\delta_{nn'}.

Note both functions sharing the same index :math:`l`.

The parameter :math:`\alpha` is an arbitrary nonnegative real number, and each value of
:math:`\alpha` gives a unique basis. This library uses the choice :math:`\alpha = 0`, because it is
clearly the simplest most straightforward choice. In this case the normalization constant reduces
to

.. math::

    N^0_{nl} = \frac{1}{2n + 3}.

This library then uses the convention in which the radial Zernike polynomials have been fully
normalized. That is, we use the Zernike functions

.. math::

    Z_{nlm}(\vec{x}) \equiv \sqrt{2n + 3}Z^0_{nlm}(\vec{x}).


