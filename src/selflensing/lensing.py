import mpmath
import numpy

from scipy import special


def ellippi(n, m):
    # https://stackoverflow.com/questions/77488171/complete-integral-of-the-third-kind-using-scipy-python-and-mathematica/77488531#77488531
    return special.elliprf(0.0, 1.0 - m, 1.0) + (n / 3.0) * special.elliprj(
        0.0, 1.0 - m, 1.0, 1.0 - n
    )


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
            * mpmath.quad(
                self._profile_f,
                [0, self.radius_unitless[sourcePlaneIdx]],
                method="gauss-legendre",
                # maxdegree=15,
            )
        )

    def magnification_derivative(self, radialCoordinate, sourcePlaneIdx, umag=None):
        if radialCoordinate < 0:
            return 0

        if umag is None:
            umag = self.uniform_magnification(sourcePlaneIdx)

        return mpmath.diff(lambda r: 0 if r < 0 else umag, radialCoordinate)

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
            * mpmath.quad(
                self._mag_profile_f,
                [0, self.radius_unitless[sourcePlaneIdx]],
                method="gauss-legendre",
                # maxdegree=15,
            )
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
