Zernike-based Radon transform
=============================

The main idea that the core of this library is built on is the fact that the dark matter direct
detection event rate can typically be expressed in terms of a Radon transform of the velocity
distribution :math:`f(\vec{v})`, defined as

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
:math:`(w,\hat{q})`. For these reasons, :math:`\hat{q}` is referred to as the *normal* and :math:`w`
as the *shell*.
