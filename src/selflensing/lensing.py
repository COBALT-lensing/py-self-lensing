import numpy
import rustlens


def wm_callback(
    offsets: numpy.ndarray, R_s_val: float, R_E_val: float
) -> numpy.ndarray:
    """
    Safe wrapper around rustlens.integrated_witt_mao_magnification.
    Offsets are passed in units of source radius (dimensionless).
    Returns unity profile on failure to keep GP stable.
    """
    # bs = offsets / source_radius
    if (
        numpy.isnan(R_E_val)
        or numpy.isnan(R_s_val)
        or numpy.any(numpy.isnan(offsets))
        or numpy.isinf(R_E_val)
        or numpy.isinf(R_s_val)
        or numpy.any(numpy.isinf(offsets))
    ):
        return numpy.ones_like(offsets)
    try:
        r = numpy.array(
            rustlens.integrated_witt_mao_magnification(
                offsets.tolist(),  # dimensionless impact parameters
                float(R_E_val),  # Einstein radius [m]
                float(R_s_val),  # source radius [m]
            )
        )
        # print("Witt & Mao result:", min(r), max(r), r)
        return r
    # except Exception:
    #     return np.ones_like(offsets)
    except BaseException:
        print("PANIC", R_E_val, R_s_val)  # , bs)
        raise
