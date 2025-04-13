Theoretical background
======================

The motivation and theory of the Zernike-based Radon transforms is described in detail in the article `arxiv:xxxx.xxxx <https://example.com>`, which introduces this method. This sections aims to give a brief introduction to the topic and the methods emplyed by this library.

In short, ZebraDM is a library that exists primarily for fast evaluation of integrals of the form

.. math::

   \overline{\mathcal{R}}[f](w,\vec{x}_0) = \int_{S^2} S(w,\hat{n})\int_{B} f(\vec{x} + \vec{x}_0)\delta(\vec{x}\cdot\hat{n} - w) \,d^3x\,d\Omega,

and also

.. math::
   
    \overline{\mathcal{R}^\perp}[f](w,\vec{x}_0) = \int_{S^2} S(w,\hat{n})\int_{B}(x^2 - (\vec{x}\cdot\hat{n})^2)f(\vec{x} + \vec{x}_0)\delta(\vec{x}\cdot\hat{n} - w) \,d^3x\,d\Omega.

The inner integral is over the unit ball :math:`B = \{\vec{x}\in\mathbb{R}^3 \mid \|\vec{x}\|\leq 1\}`, and the outer integral is over the sphere $S^2$.

These integrals are, in the context of this library called the *angle-integrated Radon* and *transverse Radon transform*, respectively. This naming is so because the inner integral of the first equation is precisely the conventional three-dimensional Radon transform of the function :math:`f(\vec{x} + \vec{x}_0)`. The functions :math:`f(\vec{x})` and :math:`S(w,\hat{n})` may, in general, be defined in coordinate systems that differ by an arbitrary rotation, although this library currently only implements solutions where they differ by a rotation about the :math:`z`-axis.

As described in more detail in the article, intergals of this form occur in computation of expected dark matter event rates in dark matter direct detection experiments, and this library has been designed for that purpose. In this context :math:`f(vec{x})` represents the dark matter velocity distribution with :math:`\vec{x}` effectively corresponding to the dark matter velocity in the laboratory frame, and :math:`\vec{x}_0` representing the velocity of the laboratory frame relative to the average motion of the dark matter. The unit vector :math:`\hat{n}` corresponds to the direction of momentum transfer in dark matter scattering, and :math:`w` corresponds to an energy parameter whose definition depends on the context. The function :math:`S(w,\hat{n})`, in turn, corresponds to a detector response function.

Zernike expansions
------------------

The core idea behind the methods of this library is the expansion of the function :math:`f(\vec{x})` in a basis of so-called Zernike functions :math:`Z_{nlm}(\vec{x})`, which are orthogonal on the unit ball. Thus

.. math::

   f(\vec{x}) = \sum_{nlm} f_{nlm}Z_{nlm}(\vec{x}).

Transformations of Zernike functions into the expansion coefficients are implemented in a companion library `zest <https://github.com/sebsassi/zest>`. Zernike functions are described in further detail in the documentation of zest. For purposes of this library, their most important property is that their Radon transform has a closed-form expression

.. math::

    \mathcal{R}[Z_{nlm}](w,\unitv{n}) = \frac{2\pi}{(n + 1)(n + 2)}(1 - w^2)C_n^{3/2}(w)Y_{lm}(\unitv{n}).

Here :math:`C_n^{3/2}(w)` are so-called Gegenbauer polynomials. It is possible to express the term :math:`(1 - w^2)C_n^{3/2}(w)` as a linear combination of two Legendre polynomials. Given this, the Radon transform of the shifted function :math:`f(\vec{x} + \vec{x}_0)` is

.. math::

    \mathcal{R}[f](w,\unitv{n}) = 2\pi\sum_{nlm}f'_{nlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\unitv{n}).

where

.. math::

    f'_{nlm} = \frac{1}{2n+3}f_{nlm} - \frac{1}{2n-1}f_{n-2,lm},

where the second term is neglected for :math:`n = 0,1` and :math:`f_{nlm} = 0` for :math:`n > L` and :math:`l > n`. This formula for :math:`f'_{nlm}` essentially defines the Radon transform in the Zernike coefficient space.

Angle integrals
---------------

Although the Radon transform formula for the Zernike coefficients is useful by itself and is also implemented in this library, a majority of this library is focused on computing integrals of the Radon transform (potentially multiplied by a response function :math:`S(w,\hat{n})`) over the directions :math:`\hat{n}`. To this end, it is important to notice that all dependence on :math:`\hat{n}` in the Radon transform is in the basis functions :math:`P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\unitv{n})`. A mild complication is the potential presence of the response function :math:`S(w,\hat{n})`. This is dealt with via the observation that we have a collection of functions

.. math::

   f'_n(\hat{n}) = \sum_{lm}f'_{nlm}Y_{lm}(\hat{n}).

These functions can be multiplied by :math:`S(w,\hat{n})`. However, as mentioned above, :math:`f'_n(\hat{n})` and :math:`S(w,\hat{n})` may be defined in coordinate systems differing by a rotation. Therefore, in practice, they first need to be rotated to a matching coordinate system. In any case, defining :math:`f^S_n(w,\hat{n})=f_n^{(R)}(\hat{n})S^{(R')}(w,\hat{n})`, where :math:`R` and :math:`R'` denote rotations applied on the functions, we end up back at

.. math::

    S(w,\hat{n})\mathcal{R}[f](w,\unitv{n}) = 2\pi\sum_{nlm}f^S_{nlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\unitv{n}).

The outcome is therefore that we only ever need to integrate

.. math::

   \int P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\unitv{n})\,d\Omega.

This integral simplifies if the integration coordinates can be chosen such that the :math:`z`-axis is in the direction of :math:`\vec{x}_0`, which requires rotation of the coefficients :math:`f^S_{nlm}`. With that the problem reduces to the evaluation of

.. math::

   A_{nl}(w,x_0) = \int_{-1}^1P_n(w + x_0z)P_l(z)\,dz,

such that

.. math::

    \overline{\mathcal{R}}[f](w,\vec{x}_0) = 2\pi\sum_{nlm}f^{S;R''}_{nlm}A_{nl}(w,x_0).

Here :math:`R''` denotes the rotation to the integration coordinates.

Integration of the transverse Radon transform proceeeds much in the same way, except there the additional term :math:`x^2 - (\vec{x}\cdot\hat{n})^2` needs to be dealt with. The problem can be reduced to integration of multiple conventional Radon transforms by means of some recursion relations of Zernike functions, after which the computation proceeds much in the same way as discussed above.

A notable fact is that in the decomposition of the transverse Radon transform to conventional Radon transforms, one of them happens to be the just :math:`\mathcal{R}[f]`. Therefore, evaluation of the transverse Radon transform of :math:`f(\vec{x})` always gives the nontransverse Radon transform for free. The library takes advantage of this fact, and so methods that evaluate the angle-integrated transverse Radon transform always return both the nontransverse, and the transverse result.
