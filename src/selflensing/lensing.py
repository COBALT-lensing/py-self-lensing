import numpy

from scipy import special, integrate, differentiate


def ellippi(n, m):
    # https://github.com/scipy/scipy/pull/15787
    # n, m = (numpy.asarray(x) for x in (n, m))
    # if numpy.any(m >= 1):
    #     raise ValueError("m must be < 1")
    y = 1 - m
    rf = special.elliprf(0, y, 1)
    rj = special.elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3


class SelfLensingSimulator(object):
    # The following implementation is based on the C++ version at:
    # https://github.com/ou-astrophysics/self-lensing-simulator

    def __init__(self, Rcomp, rE, sourcePlaneCoordinates):
        if not isinstance(rE, numpy.ndarray):
            rE = numpy.full(len(sourcePlaneCoordinates), rE)
        self.radius_unitless = Rcomp / rE
        self.radius_unitless2 = self.radius_unitless**2
        self.sourcePlaneCoordinates = sourcePlaneCoordinates

    def magnificationAtRadius(self, sourcePlaneIdx):
        kernel = (2 / self.radius_unitless[sourcePlaneIdx]) + (
            (1 + self.radius_unitless2[sourcePlaneIdx])
            / self.radius_unitless2[sourcePlaneIdx]
        ) * (
            (numpy.pi / 2)
            + numpy.arcsin(
                (self.radius_unitless2[sourcePlaneIdx] - 1)
                / (self.radius_unitless2[sourcePlaneIdx] + 1)
            )
        )

        total = kernel / numpy.pi

        return total

    def uniform_magnification(self, sourcePlaneIdx):
        sourcePlaneCoordinate = self.sourcePlaneCoordinates[sourcePlaneIdx]
        radius_unitless = self.radius_unitless[sourcePlaneIdx]
        radius_unitless2 = self.radius_unitless2[sourcePlaneIdx]
        if abs(sourcePlaneCoordinate - radius_unitless) < 1e-5:
            return self.magnificationAtRadius(sourcePlaneIdx)

        kernel1 = (sourcePlaneCoordinate - radius_unitless) ** 2
        kernel2 = numpy.sqrt(4.0 + kernel1)

        ellipticN = (
            4.0
            * radius_unitless
            * sourcePlaneCoordinate
            / ((sourcePlaneCoordinate + radius_unitless) ** 2.0)
        )

        ellipticK = numpy.sqrt(4.0 * ellipticN) / kernel2
        ellipticM = ellipticK**2

        firstTerm = ((sourcePlaneCoordinate + radius_unitless) * kernel2) / (
            2.0 * (radius_unitless2)
        )
        secondTerm = (
            (sourcePlaneCoordinate - radius_unitless)
            * (4.0 + (0.5 * (sourcePlaneCoordinate**2 - radius_unitless2)))
            / (kernel2 * radius_unitless2)
        )
        thirdTerm = (
            2.0
            * kernel1
            * (1.0 + radius_unitless2)
            / ((radius_unitless2) * (sourcePlaneCoordinate + radius_unitless) * kernel2)
        )

        kernel3 = (
            special.ellipe(ellipticM) * firstTerm
            - special.ellipk(ellipticM) * secondTerm
            + ellippi(ellipticN, ellipticM) * thirdTerm
        )

        positiveSolutionReal = (kernel3 + numpy.pi) / (2.0 * numpy.pi)
        negativeSolutionReal = (kernel3 - numpy.pi) / (2.0 * numpy.pi)
        return positiveSolutionReal + negativeSolutionReal

    def radial_profile(self, r, sourcePlaneIdx):
        LD_COEFF = 0.6
        radiusRatio = r / self.radius_unitless[sourcePlaneIdx]
        mu = numpy.sqrt(
            1.0 - (radiusRatio**2)
        )  # mu=cos(theta), theta=arcsin(radiusRatio)
        return 1.0 - LD_COEFF * (1.0 - mu)

    def _profile_f(self, r):
        return r * self.radial_profile(r, self._sourcePlaneIdx)

    def profile_integral(self, sourcePlaneIdx):
        self._sourcePlaneIdx = sourcePlaneIdx
        return (
            2
            * numpy.pi
            * integrate.fixed_quad(
                self._profile_f,
                0, self.radius_unitless[sourcePlaneIdx],
            )[0]
        )

    def magnification_derivative(self, radialCoordinate, sourcePlaneIdx, umag=None):
        if isinstance(radialCoordinate, numpy.ndarray):
            return numpy.array([self.magnification_derivative(r, sourcePlaneIdx, umag) for r in radialCoordinate])
        
        if radialCoordinate < 0:
            return 0

        if umag is None:
            umag = self.uniform_magnification(sourcePlaneIdx)

        def f(r):
            if isinstance(r, numpy.ndarray):
                if len(r.shape) == 0:
                    r = float(r)
                else:
                    return numpy.array([f(_r,) for _r in r])
            return 0 if r < 0 else umag
        return differentiate.derivative(f, radialCoordinate).df

    def _mag_profile_f(self, r):
        return r * self.radial_profile(
            r, self._sourcePlaneIdx
        ) * self._umag + 0.5 * r * self.magnification_derivative(
            r, self._sourcePlaneIdx, umag=self._umag
        )

    def magnification_profile_integral(self, sourcePlaneIdx):
        self._sourcePlaneIdx = sourcePlaneIdx
        self._umag = self.uniform_magnification(sourcePlaneIdx)
        return (
            2
            * numpy.pi
            * integrate.fixed_quad(
                self._mag_profile_f,
                0, self.radius_unitless[sourcePlaneIdx],
            )[0]
        )

    def magnification(self, sourcePlaneIdx):
        return self.magnification_profile_integral(
            sourcePlaneIdx
        ) / self.profile_integral(sourcePlaneIdx)

    def magnifications(self):
        return [
            float(self.magnification(sourcePlaneIdx))
            for sourcePlaneIdx in range(len(self.sourcePlaneCoordinates))
        ]


def heyrovsky_magnification(l, r, Re, Rstar):
    """
    Computes the angle-integrated magnification of the source disk at radius r
    when the lens is at l distance from the disk centre.
    """
    if isinstance(r, numpy.ndarray):
        return numpy.array([heyrovsky_magnification(l, _r, Re, Rstar) for _r in r])
    l, r, Re, Rstar = map(float, (l, r, Re, Rstar))
    epsilon = Re / Rstar
    delta_limit = 1e-5
    delta = l - r
    mod_delta = abs(delta)

    A = 0

    if mod_delta > delta_limit:
        p = numpy.sqrt(l * r / ((l - r) ** 2 + 4 * epsilon**2))
        k = 4 * epsilon * p / (l + r)
        m = k**2
        phi = 4 * l * r / (l + r) ** 2
        int1 = special.ellipk(m)
        phi = -1 * phi
        int2 = ellippi(phi, m)

        prefix = 4 / ((l + r) * numpy.sqrt((l - r) ** 2 + 4 * epsilon**2))
        A = prefix * (2 * epsilon**2 * int1 + (l - r) ** 2 * int2)

    elif delta >= 0:
        delta1 = (
            (2 * epsilon / l)
            * (1 - (delta / (2 * l)))
            * numpy.log(
                (8 * epsilon * l)
                / (numpy.sqrt(delta**2) * numpy.sqrt((l**2) + (epsilon**2)))
            )
        )
        delta2 = 4 * numpy.atan(l / epsilon) + delta * epsilon * (
            2 * (l**2) + (epsilon**2)
        ) / ((l**2) * ((l**2) + (epsilon**2)))

        A = delta1 + delta2

    return A / (2 * numpy.pi)


def integrated_luminosity(A, B):
    return float(integrate.fixed_quad(lambda r: B(r) * A(r) * r, 0, 1)[0])


def ld_linear(r):
    LD_COEFF = 0.6
    return 1.0 - LD_COEFF * (1.0 - numpy.sqrt(1.0 - (r**2)))


def uniform_luminosity(A, L):
    return A * L


def witt_mao_magnification(l, Re, Rstar):
    r = Rstar / Re
    r2 = r**2
    l *= r

    if abs(l - r) < 1e-5:
        kernel = (2 / r) + ((1 + r2) / r2) * (
            (numpy.pi / 2) + numpy.arcsin((r2 - 1) / (r2 + 1))
        )
        return kernel / numpy.pi

    kernel1 = (l - r) ** 2
    kernel2 = numpy.sqrt(4.0 + kernel1)

    ellipticN = 4.0 * r * l / ((l + r) ** 2.0)

    ellipticK = numpy.sqrt(4.0 * ellipticN) / kernel2
    ellipticM = ellipticK**2

    firstTerm = ((l + r) * kernel2) / (2.0 * (r2))
    secondTerm = (l - r) * (4.0 + (0.5 * (l**2 - r2))) / (kernel2 * r2)
    thirdTerm = 2.0 * kernel1 * (1.0 + r2) / ((r2) * (l + r) * kernel2)

    kernel3 = (
        special.ellipe(ellipticM) * firstTerm
        - special.ellipk(ellipticM) * secondTerm
        + ellippi(ellipticN, ellipticM) * thirdTerm
    )

    positiveSolutionReal = (kernel3 + numpy.pi) / (2.0 * numpy.pi)
    negativeSolutionReal = (kernel3 - numpy.pi) / (2.0 * numpy.pi)
    return positiveSolutionReal + negativeSolutionReal
