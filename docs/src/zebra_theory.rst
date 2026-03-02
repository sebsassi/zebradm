Radon transfomr
===============

The main motivation for this library is the observation that the dark matter direct detection
event rate can typically be expressed in terms of a Radon transform of the velocity distribution
:math:`f(\vec{v})`, defined as

.. math::

    \mathcal{R}[f](\tilde{w},\hat{q})
        = \int \delta(\tilde{w} - \vec{v}\cdot\hat{q})f(\vec{v} + \vec{v}_\text{lab}(t))\,d^3v.

Here :math:`\hat{q}` denotes a unit vector, which correspond to a direction of momentum transfer,
and :math:`\tilde{w}` is an energy parameter for which, for the purposes of this discussion, a
sufficient condition is that it is independent of the dark matter velocity :math:`\vec{v}`. The lab
velocity :math:`\vec{v}_\text{lab}(t)` parametrizes the fact that the coordinate system where the
dark matter scattering is observed may be moving relative to the dark matter distribution at some
time-dependent velocity.

In the most general case, it turns out that a pure Radon transform of the velocity distribution
is not sufficient for describing all potential types of dark matter scattering. More generally,
the Radon transform is of the form

.. math::

    \mathcal{R}_K[f](\tilde{w},\hat{q})
        = \int \delta(\tilde{w} - \vec{x}\cdot\hat{q})K(\vec{v},\hat{q})f(\vec{v} + \vec{v}_\text{lab}(t))\,d^3v,

where :math:`K(\vec{v},\hat{q})` denotes an extra kernel function. For example, a common case is
:math:`K(\vec{v},\hat{q}) = \|\vec{v}_\perp\|^2`, with :math:`\vec{v}_\perp = \vec{v} -
(\vec{v}\cdot\hat{q})\hat{q}`, which defines the so-called transverse Radon transform. In practice,
all kernels are at most quadratic in the components of :math:`\vec{v}`.

Regardless of the presence of a kernel, the above expression is still ultimately just a Radon
transform. We will therefore move on with discussion of the evaluation of the basic Radon transform
of just the velocity distribution, and later describe how this solution can be applied in the more
general case.

The first step is to make a change of variables to the integral to transform to coordinates where
the distribution :math:`f(\vec{v})` has no overall time-dependent motion, such that

.. math::

    \mathcal{R}[f](\tilde{w} + \vec{v}_\text{lab}(t)\cdot\hat{q},\hat{q})
        = \int \delta(\tilde{w} + \vec{v}_\text{lab}(t)\cdot\hat{q} - \vec{v}\cdot\hat{q})f(\vec{v})\,d^3v.

Next, we assume that :math:`f(\vec{v})` is supported within a ball of radius :math:`v_\text{max}`.
This assumption does not meaninfully restrict what kinds of distributions can be described, because
the basic assumption is that the dark matter halo is cold, i.e., that the dark matter is
broadly nonrelativistic, which means that all dark matter can be approximated as having speeds less
than :math:`v_\text{max}`. Typically, assuming that the velocity distribution is (nearly)
virialized, this can be taken to be the escape velocity: :math:`v_\text{max} = v_\text{esc}`. A
change of variables to :math:`\vec{x} = \vec{v}/v_\text{max}` then takes this to a form

.. math::

   \mathcal{R}[f](\tilde{w},\hat{q}) = v_{max}^2\mathcal{R}[f](w,\hat{q}),

where

.. math::

    \mathcal{R}[f](w + \vec{x}_\text{off}\cdot\hat{q},\hat{q})
        = \int_B \delta(w + \vec{x}_\text{off}\cdot\hat{q} - \vec{x}\cdot\hat{q})f(\vec{x})\,d^3x

is a Radon transform over the unit ball :math:`B = {\vec{x} \in \mathbb{R}^3 : \|\vec{x}\| \leq 1}`
with an offset :math:`\vec{x}_\text{off} = \vec{x}_\text{lab}/v_\text{max}`. This abstract form of
the Radon transform is used for describing Radon transforms in the library API, because it
describes a solution to a much more general problem than that of dark matter scattering.

Geometrically, the 3D Radon transform describes an integral of a function
:math:`f(\vec{x} + \vec{x}_\text{off})` over a plane defined by :math:`\hat{q}` and :math:`w`. The
vector :math:`\vec{x}_\text{off}` is therfore called the *offset*, because it offsets the function
being integrated. The plane of integration is such that its normal is in the direction of
:math:`\hat{q}`, and its distance from the origin (that, is the distance of its closest point) is
:math:`w`. Hence :math:`\hat{q}` is referred to as the *normal*. The plane is uniquely described by
these two parameters. Furthermore, for a given :math:`w > 0`, there is a bijective mapping from the
sphere :math:`S^2` onto a spherical shell of radius :math:`w`, given by :math:`(w,\hat{q}) \mapsto
w\hat{q}`, which is the point closest to the origin of the plane corresponding to
:math:`(w,\hat{q})`. For these reasons, we refer to :math:`\hat{q}` as the *normal* and :math:`w`
as the *shell*.

Zernike-based Radon transform
-----------------------------

The core algorithm in this library is based on the fact that the Radon transform of the Zernike
functions has a closed form solution. Namely, given the (radially unnormalized) Zernike function
:math:`Z^\alpha_{nlm}(\vec{x})`, its radon transform is given by

.. math::

    \mathcal{R}[Z^\alpha_{nlm}](w,\hat{q})
        = \frac{1}{c^\alpha_{nl}}(1 - w^2)^{\alpha + 1} C^{\alpha + 3/2}_n(w)Y_{lm}(\hat{q}),

where :math:`C^{\alpha + 3/2}_n(w)` are Gegenbauer polynomials, and the factor :math:`c^\alpha_{nl}`
is given by

.. math::

    c^\alpha_{nl}
        = \frac{2^{-2(1 + \alpha)}}{\sqrt{\pi}((n - l - 2)/2)_\alpha}
            \frac{\Gamma(n + 2\alpha + 3)}{\Gamma(n + 1)\Gamma(\alpha + 3/2)}.

For the family with :math:`\alpha = 0` used by this library, the above expression reduces to

.. math::

    \mathcal{R}[Z^0_{nlm}](w,\hat{q})
        = 2\pi\frac{1 - w^2}{(n + 1)(n + 2)}C^{3/2}_n(w)Y_{lm}(\hat{q}).

The Gegenbauer polynomials :math:`C^{3/2}_n(w)` here are related to the Legendre polynomials via

.. math::

    \frac{1 - w^2}{(n + 1)(n + 1)}C^{3/2}_n(w) = \frac{P_n(w) - P_{n + 2}(w)}{2n + 3}.

We can therefore immediately write down an expression for a function expressed as a linear
combination of radially normalized Zernike polynmials

.. math::

    f(\vec{x}) = \sum_{n = 0}^N\sum_{\substack{l = 0\\ 2 \mid l}}^n\sum_{|m| \leq l} f_{nlm}Z_{nlm}(\vec{x})

as

.. math::

    \mathcal{R}[f](w,\hat{q})
        = \sum_{n = 0}^N\sum_{\substack{l = 0\\ 2 \mid l}}^n\sum_{|m| \leq l} \hat{f}_{nlm}P_n(w)Y_{lm}(\hat{q}),

where the Radon transform coefficients :math:`\hat{f}_{nlm}` are given by

.. math::

    \hat{f}_{nlm} = \frac{f_{nlm}}{\sqrt{2n + 3}} - \frac{f_{n - 2,lm}}{\sqrt{2n - 1}}.

Weighted angle-integrated Radon transform
-----------------------------------------

The object of interest for most common dark matter event rate computations is the weighted
angle-integrated Radon transform

.. math::

    \overline{\mathcal{R}}_W[f](w,\vec{x}_0)
        \equiv \int_{S^2}W(w,\hat{q})\mathcal{R}[f](w + \vec{x}_0\cdot\hat{q},\hat{q})\,d\Omega.
