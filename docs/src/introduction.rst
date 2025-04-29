Theory -- Zernike-based Radon transforms
========================================

The motivation and theory of the Zernike-based Radon transforms is described in detail in the
article `arxiv:2504.19714 <https://arxiv.org/abs/2504.19714>`_, which introduces this method. This sections aims
to give a brief introduction to the topic and the methods emplyed by this library.

In short, ZebraDM is a library that exists primarily for fast evaluation of integrals of the form

.. math::

   \overline{\mathcal{R}}[f](w,\vec{x}_0)
      = \int_{S^2} S(w,\hat{n})\int_{B} f(\vec{x} + \vec{x}_0)\delta(\vec{x}\cdot\hat{n} - w) \,d^3x\,d\Omega,

and also

.. math::
   
    \overline{\mathcal{R}^\perp}[f](w,\vec{x}_0)
      = \int_{S^2} S(w,\hat{n})\int_{B}(x^2 - (\vec{x}\cdot\hat{n})^2)f(\vec{x} + \vec{x}_0)\delta(\vec{x}\cdot\hat{n} - w) \,d^3x\,d\Omega.

The inner integral is over the unit ball

.. math::

   B = \{\vec{x}\in\mathbb{R}^3 \mid \|\vec{x}\|\leq 1\},

and the outer integral is over the sphere :math:`S^2`.

These integrals are, in the context of this library called the *angle-integrated Radon* and
*transverse Radon transform*, respectively. This naming is so because the inner integral of the
first equation is precisely the conventional three-dimensional Radon transform of the function
:math:`f(\vec{x} + \vec{x}_0)`. The parameter :math:`\vec{x}_0` here is an arbitrary offset applied
on the distribution. The functions :math:`f(\vec{x})` and :math:`S(w,\hat{n})` may, in general, be
defined in coordinate systems that differ by an arbitrary rotation, although this library currently
only implements solutions where they differ by a rotation about the :math:`z`-axis.

As described in more detail in the article, intergals of this form occur in computation of expected
dark matter event rates in dark matter direct detection experiments. In this context :math:`f(\vec{x})`
represents the dark matter velocity distribution with :math:`\vec{x}` effectively corresponding to
the dark matter velocity in the laboratory frame, and :math:`\vec{x}_0` representing the velocity
of the laboratory frame relative to the average motion of the dark matter. The unit vector
:math:`\hat{n}` corresponds to the direction of momentum transfer in dark matter scattering, and
:math:`w` corresponds to an energy parameter whose definition depends on the context. The function
:math:`S(w,\hat{n})`, in turn, corresponds to a detector response function.

Radon transforms
----------------

The Radon transform is a map, which maps a function :math:`f` defined on :math:`\mathbb{R}^n` onto
the space of :math:`(n-1)`-dimensional hyperplanes on :math:`\mathbb{R}^n`. In 3D, this means
mapping the function :math:`f` onto the space of planes on :math:`\mathbb{R}^3`. The Radon
transform in this case is given by the integral formula

.. math::

   \mathcal{R}[f](w,\hat{n}) = \int_{\mathbb{R}^3} \delta(\vec{x}\cdot\hat{n} - w)f(\vec{x})\,d^3x,

where :math:`\delta` is the Dirac delta-function. The delta-function forces the integration to be
over a plane whose normal vector is :math:`\hat{n}`, and whose distance from the origin is
:math:`w`. These two parameters together uniquely define a plane in :math:`\mathbb{R}^3`.

The function :math:`f` doesn't have to be supported on entirity of :math:`\mathbb{R}^3`. If
:math:`f` is nonzero only on some measurable subset :math:`U \subset \mathbb{R}^3`, the we may
define

.. math::

   \mathcal{R}[f](w,\hat{n}) = \int_U \delta(\vec{x}\cdot\hat{n} - w)f(\vec{x})\,d^3x.

This library, in particular, assumes that :math:`f` is zero outside the unit ball. For practical
purposes in terms of computation with finite precision floating point numbers, this is not much of
a restriction at all. For example, any square-integrable function on :math:`\mathbb{R}^3` can be
clamped to zero beyond some finite radius :math:`R` such that its Radon transform will remain the
same up to some precision. Then the coordinates can be rescaled such that :math:`R\rightarrow 1`.
Most numerical problems in :math:`\mathbb{R}^3` can therefore be restricted to a ball without
losses, and in terms of functions supported on a ball, the rescaling lets us assume a unit ball
without loss of generality.

Since this library deals with angle-integrated Radon transforms, i.e., interals of the Radon
transform over the unit vectors :math:`\hat{n}`, it is useful to think of the planes as the tangent
planes of spherical shells of radius :math:`w` at the points :math:`w\hat{n}`. Then the
angle-integrated Radon transform maps :math:`f` onto these spherical shells parametrized
by :math:`w`. Due to this identification, the parameter :math:`w` is known as the *shell*
parameter, or ``shell`` for short in the code.

An important property of Radon transforms is that if we use :math:`f` to define a new offset
function :math:`f_{\vec{x}_0}(\vec{x})=f(\vec{x}+\vec{x}_0)`, then

.. math::

    \mathcal{R}[f_{\vec{x}_0}](w,\hat{n})=\mathcal{R}[f](w+\vec{x}_0\cdot\hat{n},\hat{n}).

Therefore handling Radon transforms of functions with arbitrary offsets is straightforward to deal
with.

It is worth noting that if :math:`f` is zero outside of the unit ball, then the Radon transform is
zero for :math:`w > 1`, because none of the planes tangent to the outer shells intersect the unit
ball. If we take into account the arbitrary offset :math:`\vec{x}_0`, then this implies that the
Radon transform is zero for :math:`|w + \vec{x}_0\cdot\hat{n}| > 1`. This means that the
angle-integrated Radon transform is nonzero only for the shell parameters :math:`w \leq 1 + x_0`,
where :math:`x_0` is the length of :math:`\vec{x}_0`.

Zernike expansions
------------------

The core idea behind the methods of this library is the expansion of the function :math:`f(\vec{x})`
in a basis of so-called Zernike functions :math:`Z_{nlm}(\vec{x})`, which are orthogonal on the
unit ball. Thus

.. math::

    f(\vec{x}) = \sum_{nlm} f_{nlm}Z_{nlm}(\vec{x}).

Transformations of Zernike functions into the expansion coefficients are implemented in a companion
library `zest <https://github.com/sebsassi/zest>`_. Zernike functions are described in further
detail in the documentation of zest. For purposes of this library, their most important property is
that their Radon transform has a closed-form expression

.. math::

    \mathcal{R}[Z_{nlm}](w,\hat{n}) = \frac{2\pi}{(n + 1)(n + 2)}(1 - w^2)C_n^{3/2}(w)Y_{lm}(\hat{n}).

Here :math:`C_n^{3/2}(w)` are so-called Gegenbauer polynomials. It is possible to express the term
:math:`(1 - w^2)C_n^{3/2}(w)` as a linear combination of two Legendre polynomials. Given this, the
Radon transform of the shifted function :math:`f(\vec{x} + \vec{x}_0)` is

.. math::

    \mathcal{R}[f](w,\hat{n}) = 2\pi\sum_{nlm}f'_{nlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).

where

.. math::

    f'_{nlm} = \frac{1}{2n+3}f_{nlm} - \frac{1}{2n-1}f_{n-2,lm},

where the second term is neglected for :math:`n = 0,1` and :math:`f_{nlm} = 0` for :math:`n > L`
and :math:`l > n`. This formula for :math:`f'_{nlm}` essentially defines the Radon transform in the
Zernike coefficient space.

Angle integrals
---------------

Although the Radon transform formula for the Zernike coefficients is useful by itself and is also
implemented in this library, a majority of this library is focused on computing integrals of the
Radon transform (potentially multiplied by a response function :math:`S(w,\hat{n})`) over the
directions :math:`\hat{n}`. To this end, it is important to notice that all dependence on
:math:`\hat{n}` in the Radon transform is in the basis functions

.. math::

    P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).

A mild complication is the potential presence of the response function :math:`S(w,\hat{n})`. This
is dealt with via the observation that we have a collection of functions

.. math::

   f'_n(\hat{n}) = \sum_{lm}f'_{nlm}Y_{lm}(\hat{n}).

These functions can be multiplied by :math:`S(w,\hat{n})`. However, as mentioned above,
:math:`f'_n(\hat{n})` and :math:`S(w,\hat{n})` may be defined in coordinate systems differing by a
rotation. Therefore, in practice, they first need to be rotated to a matching coordinate system. In
any case, defining

.. math::

    f^S_n(w,\hat{n}) = f_n^{(R)}(\hat{n})S^{(R')}(w,\hat{n}),

where :math:`R` and :math:`R'` denote rotations applied on the functions, we end up back at

.. math::

    S(w,\hat{n})\mathcal{R}[f](w,\hat{n}) = 2\pi\sum_{nlm}f^S_{nlm}P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n}).

The outcome is therefore that we only ever need to integrate

.. math::

   \int P_n(w + \vec{x}_0\cdot\hat{n})Y_{lm}(\hat{n})\,d\Omega.

This integral simplifies if the integration coordinates can be chosen such that the :math:`z`-axis
is in the direction of :math:`\vec{x}_0`, which requires rotation of the coefficients
:math:`f^S_{nlm}`. With that the problem reduces to the evaluation of

.. math::

   A_{nl}(w,x_0) = \int_{-1}^1P_n(w + x_0z)P_l(z)\,dz,

such that

.. math::

    \overline{\mathcal{R}}[f](w,\vec{x}_0) = 2\pi\sum_{nlm}f^{S;R''}_{nlm}A_{nl}(w,x_0).

Here :math:`R''` denotes the rotation to the integration coordinates.

Integration of the transverse Radon transform proceeeds much in the same way, except there the
additional term :math:`x^2 - (\vec{x}\cdot\hat{n})^2` needs to be dealt with. The problem can be
reduced to integration of multiple conventional Radon transforms by means of some recursion
relations of Zernike functions, after which the computation proceeds much in the same way as
discussed above.

A notable fact is that in the decomposition of the transverse Radon transform to conventional Radon
transforms, one of them happens to be the just :math:`\mathcal{R}[f]`. Therefore, evaluation of the
transverse Radon transform of :math:`f(\vec{x})` always gives the nontransverse Radon transform for
free. The library takes advantage of this fact, and so methods that evaluate the angle-integrated
transverse Radon transform always return both the nontransverse, and the transverse result.
