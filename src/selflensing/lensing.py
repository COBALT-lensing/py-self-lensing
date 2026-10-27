import numpy
import rustlens
import jax

from jax import numpy as jnp
from scipy import special


# Fixed linear limb-darkening coefficient (Eddington grey atmosphere).
LD_COEFF = 0.6
LD_LINEAR_INT = 3.141592653589793 * (1.0 - LD_COEFF / 3.0)
AREA_QUADRATURE_POINTS = 96
BRIGHTNESS_QUADRATURE_POINTS = 96
# Bound the timestamp x brightness x angle workspace, including under vmap
# over MCMC walkers. This changes evaluation order, not quadrature resolution.
LENSING_TIME_BATCH_SIZE = 32

_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = special.roots_legendre(
    AREA_QUADRATURE_POINTS
)
_THETA = numpy.asarray(0.25 * numpy.pi * (_LEGENDRE_NODES + 1.0))
_THETA_WEIGHTS = numpy.asarray(0.25 * numpy.pi * _LEGENDRE_WEIGHTS)
_BRIGHTNESS_NODES, _BRIGHTNESS_WEIGHTS = special.roots_legendre(
    BRIGHTNESS_QUADRATURE_POINTS
)
_BRIGHTNESS_R = numpy.asarray(0.5 * (_BRIGHTNESS_NODES + 1.0))
_BRIGHTNESS_WEIGHTS = numpy.asarray(0.5 * _BRIGHTNESS_WEIGHTS)


def wm_callback(
    offsets: numpy.ndarray, R_s_val: float, R_E_val: float
) -> numpy.ndarray:
    """
    Safe wrapper around rustlens.integrated_witt_mao_magnification.
    Offsets are passed in units of source radius (dimensionless).
    Zero Einstein radius returns unity; invalid geometry returns NaN.
    """
    return native_magnification_callback(offsets, R_s_val, R_E_val)


def native_magnification_callback(offsets, rstar, re, method="integrated_witt_mao_magnification"):
    """Evaluate RustLens with scalar or per-sample radii and foreground gating.

    RustLens accepts one Einstein radius per call. The scalar-radius fast path
    remains available; varying radii use individual calls with identical units.
    Zero RE means no foreground lensing. Other invalid geometries return NaN.
    """
    offsets, rstar, re = numpy.broadcast_arrays(offsets, rstar, re)
    valid = numpy.isfinite(offsets) & numpy.isfinite(rstar) & numpy.isfinite(re) & (offsets>=0) & (rstar>0) & (re>=0)
    result = numpy.full(offsets.shape, numpy.nan, dtype=numpy.result_type(offsets, numpy.float32))
    result[valid & (re==0)] = 1.
    active = valid & (re>0)
    if not numpy.any(active):
        return result
    function = getattr(rustlens, method)
    if numpy.all(re[active]==re[active][0]) and numpy.all(rstar[active]==rstar[active][0]):
        result[active] = function(offsets[active].tolist(), float(re[active][0]), float(rstar[active][0]))
    else:
        for index in zip(*numpy.nonzero(active)) if result.ndim else [()]:
            result[index] = function([float(offsets[index])], float(re[index]), float(rstar[index]))[0]
    return result


def _theta_grid(dtype):
    return jnp.asarray(_THETA, dtype=dtype), jnp.asarray(_THETA_WEIGHTS, dtype=dtype)


def _brightness_grid(dtype):
    return (
        jnp.asarray(_BRIGHTNESS_R, dtype=dtype),
        jnp.asarray(_BRIGHTNESS_WEIGHTS, dtype=dtype),
    )


def _disk_area_magnification(l, epsilon):
    """Equivalent Witt–Mao uniform-disk integral in lens-centered coordinates.

    The radial antiderivative of s*A_point(s) is s*hypot(s, 2*epsilon)/2.
    Integrating it between the ray/disk intersections avoids cancellation of
    separately quadrature-approximated elliptic terms near the limb. All lengths
    here use the disk radius. No interval is replaced by its exact-limb value.
    """
    theta, weights = _theta_grid(l.dtype)
    l, epsilon = l[..., None], epsilon[..., None]

    # Interior lens: pair rays at theta and pi-theta. Cluster nodes at pi/2
    # with theta = pi/2 - (pi/2)*t^2 to resolve the near-limb boundary layer.
    t = theta / (0.5 * jnp.pi)
    inside_l = jnp.minimum(l, 1.0)
    center = inside_l * jnp.sin(0.5*jnp.pi*t**2)
    gap = (1.0 - inside_l) * (1.0 + inside_l)
    half_chord = jnp.sqrt(gap + center**2)
    upper = half_chord + center
    # Rationalized half_chord-center, safe even at the limb.
    lower = gap / upper
    interior = jnp.sum(
        weights * t * (upper*jnp.hypot(upper, 2.0*epsilon)
                       + lower*jnp.hypot(lower, 2.0*epsilon)), axis=-1
    )

    # Exterior lens: sin(theta_ray)=sin(theta)/l removes the square-root
    # endpoint. Rationalize H(upper)-H(lower) before multiplying by dtheta_ray.
    outside_l = jnp.maximum(l, 1.0)
    chord = jnp.cos(theta)
    center = jnp.sqrt((outside_l-1.0)*(outside_l+1.0) + chord**2)
    upper = center + chord
    lower = (outside_l-1.0)*(outside_l+1.0) / upper
    exterior = jnp.sum(
        weights * 2.0*chord**2 * (upper**2+lower**2+4.0*epsilon**2)
        / (upper*jnp.hypot(upper, 2.0*epsilon)
           + lower*jnp.hypot(lower, 2.0*epsilon)), axis=-1
    )
    return (2.0/jnp.pi) * jnp.where(l[..., 0] <= 1.0, interior, exterior)


def _geometry(l, re, rstar):
    # Integer lists must not truncate radii. Mask invalid inputs *before*
    # evaluation, so an inactive invalid branch cannot poison valid gradients.
    l, re, rstar = jnp.asarray(l), jnp.asarray(re), jnp.asarray(rstar)
    dtype = jnp.result_type(l, re, rstar, 1.0)
    l, re, rstar = jnp.broadcast_arrays(
        jnp.asarray(l, dtype=dtype), jnp.asarray(re, dtype=dtype),
        jnp.asarray(rstar, dtype=dtype),
    )
    valid = (jnp.isfinite(l) & (l >= 0.0) & jnp.isfinite(re) & (re > 0.0)
             & jnp.isfinite(rstar) & (rstar > 0.0))
    return (jnp.where(valid, l, 0.0), jnp.where(valid, re, 1.0),
            jnp.where(valid, rstar, 1.0), valid)


def witt_mao_magnification_jax(l, re, rstar):
    """Uniform-disc magnification; ``l=d/Rstar``, radii in matching units.

    Like RustLens, this interface takes separation in source radii, not
    Einstein radii. Both physical radii must use the same projected plane.
    """
    l, re, rstar, valid = _geometry(l, re, rstar)
    formula_value = _disk_area_magnification(l, re/rstar)
    return jnp.where(valid, formula_value, jnp.nan)


def integrated_witt_mao_magnification_jax(l, re, rstar):
    """Linear limb-darkened magnification, coefficient 0.6; ``l=d/Rstar``.

    Integrates B(r) d[r^2 A_disk(r)] by parts. The substitution
    mu=sqrt(1-r^2) removes the brightness derivative's limb singularity;
    splitting at r=l resolves the disk kernel's contact with the lens.
    """
    l, re, rstar, valid = _geometry(l, re, rstar)
    shape = l.shape
    nodes, weights = _brightness_grid(l.dtype)

    def integrate_one(inputs):
        offset, epsilon = inputs
        # At an exterior lens there is no interior singularity. Keep the
        # inactive square root away from zero for finite parameter gradients.
        interior = offset < 1.0
        cut = jnp.where(interior, jnp.sqrt(jnp.where(interior, 1.0-offset**2, 1.0)), 0.0)

        def interval(a, b):
            # Cluster at both endpoints; this also smooths the central
            # sqrt(1-mu) limit. Compute 1-mu without subtracting near-equals.
            sine = jnp.sin(0.5*jnp.pi*nodes)
            cosine = jnp.cos(0.5*jnp.pi*nodes)
            radius2 = ((1.0-b)+(b-a)*cosine**2) * ((1.0+a)+(b-a)*sine**2)
            radius = jnp.sqrt(jnp.where(b > a, radius2, 0.25))
            cumulative = radius**2 * _disk_area_magnification(offset/radius, epsilon/radius)
            jacobian = jnp.pi*(b-a)*sine*cosine
            return jnp.sum(weights*jacobian*cumulative)

        mu_int = interval(0.0, cut) + interval(cut, 1.0)
        full_flux = _disk_area_magnification(offset, epsilon)
        return ((1.0-LD_COEFF)*full_flux + LD_COEFF*mu_int)/(1.0-LD_COEFF/3.0)

    magnification = jax.lax.map(
        integrate_one, (l.ravel(), (re/rstar).ravel()), batch_size=LENSING_TIME_BATCH_SIZE
    ).reshape(shape)
    return jnp.where(valid, magnification, jnp.nan)
