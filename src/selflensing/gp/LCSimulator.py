import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# ----------------------------------------------------------------------------
# Boolean flag generators
# ----------------------------------------------------------------------------

def assign_flags(N: int, seed: int = None) -> tuple:
    """
    Create boolean arrays: lensing, ellipsoidal, flare.
    """
    rng = np.random.RandomState(seed)

    # 1) Half of all N are “lensing” (flag=1), rest are 0
    lensing = np.zeros(N, dtype=int)
    lens_idx = rng.choice(N, N // 2, replace=False)
    lensing[lens_idx] = 1

    # 2) Of those lensing, half (floor division) are “ellipsoidal”
    ellipsoidal = np.zeros(N, dtype=int)
    ellip_subset_size = len(lens_idx) // 2
    ellip_idx = rng.choice(lens_idx, ellip_subset_size, replace=False)
    ellipsoidal[ellip_idx] = 1
    
    # 3) “Flare” flags are assigned among those NOT flagged as ellipsoidal
    flare = np.zeros(N, dtype=int)
    non_ellip_idx = np.where(ellipsoidal == 0)[0]
    flare_subset_size = N // 3
    # If N//3 > len(non_ellip_idx), it will error; you may wish to guard that.
    flare_idx = rng.choice(non_ellip_idx, flare_subset_size, replace=False)
    flare[flare_idx] = 1
    return lensing, ellipsoidal, flare

def generate_kernels(N: int, seed: int = None) -> pd.DataFrame:
    """
    Generate DataFrame of N random kernel_type strings and flags.
    """
    lensing, ellipsoidal, flare = assign_flags(N, seed)
    records = []
    for i in range(N):
        kt = "GS"
        if ellipsoidal[i]:
            kt += "E"
        if flare[i]:
            kt += "F"
        records.append(kt)
    arr = np.array([records, lensing, ellipsoidal, flare]).T
    return pd.DataFrame(arr, columns=["kernel_type", "lensing", "ellipsoidal", "flare"])

# File: LCSimulator.py
def safe_gp_sample(gp, key=None):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
    try:
        draw = gp.sample(key)
        if jnp.any(~jnp.isfinite(draw)):
            mask = ~jnp.isfinite(draw)
            draw = draw.at[mask].set(gp.mean[mask])
        return draw
    except Exception:
        return gp.mean