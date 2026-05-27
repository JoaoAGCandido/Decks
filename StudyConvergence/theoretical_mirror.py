"""Theoretical no-drift guiding-center motion in a magnetic mirror field.

The magnetic field is

    B(x, y, z) = (-a*x*z, -a*y*z, B0 + a*z**2)

with E = 0.  The guiding center is constrained to a magnetic field line,
the magnetic moment per unit rest mass is conserved, and the dynamics use
proper velocity u = gamma*v.  The default units are normalized so c = 1.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, interpolate, optimize


__all__ = ["theoretical_mirror_position"]


def _as_vector3(name, value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector; got shape {vector.shape}.")
    return vector


def _as_time_array(t):
    scalar_input = np.ndim(t) == 0
    t_array = np.atleast_1d(np.asarray(t, dtype=float))
    if t_array.ndim != 1:
        raise ValueError(f"t must be a scalar or a 1D array; got shape {t_array.shape}.")
    return t_array, scalar_input


def _S(z, a, B0):
    return B0 + a * np.asarray(z) ** 2


def _Bmag(z, K, a, B0):
    z = np.asarray(z)
    S = _S(z, a, B0)
    return np.sqrt(S * S + a * a * K * z * z / S)


def _dBmag_dz(z, K, a, B0):
    S = _S(z, a, B0)
    B = _Bmag(z, K, a, B0)
    return z * (2.0 * a * S + a * a * K * B0 / (S * S)) / B


def _mirror_B_vector(R, a, B0):
    x, y, z = R
    return np.array([-a * x * z, -a * y * z, B0 + a * z * z], dtype=float)


def _positions_from_z(z, K, phi0, a, B0):
    z = np.asarray(z, dtype=float)
    r = np.sqrt(K / _S(z, a, B0))
    return np.column_stack((r * np.cos(phi0), r * np.sin(phi0), z))


def _clip_upar_sq(upar_sq, scale, atol):
    values = np.asarray(upar_sq, dtype=float)
    tol = max(1000.0 * np.finfo(float).eps * max(scale, 1.0), 100.0 * atol)
    if np.any(values < -tol):
        min_value = np.min(values)
        raise ValueError(
            "upar_sq became negative inside the allowed mirror interval "
            f"(minimum {min_value:.6e})."
        )
    return np.maximum(values, 0.0)


def _find_positive_mirror_z(z0, B_initial, upar0, muhat, K, a, B0, rtol, atol):
    if muhat <= 0.0:
        raise ValueError("muhat must be positive to find magnetic mirror points.")

    B_target = B_initial + upar0 * upar0 / (2.0 * muhat)
    z_low = abs(z0)

    def residual(z):
        return float(_Bmag(z, K, a, B0) - B_target)

    f_low = residual(z_low)
    B_tol = max(100.0 * np.finfo(float).eps * max(B_target, 1.0), 100.0 * atol)
    if abs(f_low) <= B_tol:
        return z_low, B_target
    if f_low > 0.0:
        raise ValueError(
            "Initial state is outside the allowed mirror interval: "
            "Bmag(abs(z0)) is larger than the mirror target."
        )

    z_high = max(1.0, 1.25 * z_low + 1.0)
    f_high = residual(z_high)
    for _ in range(256):
        if f_high >= 0.0:
            break
        z_high = 2.0 * z_high + 1.0
        f_high = residual(z_high)
    else:
        raise ValueError("Could not bracket the positive mirror point.")

    zm = optimize.brentq(
        residual,
        z_low,
        z_high,
        xtol=max(atol, 1.0e-14),
        rtol=max(rtol, 4.0 * np.finfo(float).eps),
    )
    return float(zm), B_target


def _endpoint_theta_limit(zm, K, muhat, Gamma, a, B0):
    if zm <= 0.0 or muhat <= 0.0:
        return np.nan
    S_m = float(_S(zm, a, B0))
    B_m = float(_Bmag(zm, K, a, B0))
    dB_dz_m = abs(float(_dBmag_dz(zm, K, a, B0)))
    if dB_dz_m <= 0.0:
        return np.nan
    return Gamma * B_m / S_m * np.sqrt(zm / (muhat * dB_dz_m))


def _theta_integrand(theta, zm, K, B_mirror, muhat, Gamma, a, B0, atol):
    theta = np.asarray(theta, dtype=float)
    z = zm * np.sin(theta)
    cos_theta = np.cos(theta)
    S = _S(z, a, B0)
    B = _Bmag(z, K, a, B0)
    S_m = float(_S(zm, a, B0))
    B_m = float(_Bmag(zm, K, a, B0))
    # Use B_m**2 - B**2 factored in z**2 to avoid endpoint cancellation.
    w_gap = zm * zm * cos_theta * cos_theta
    q_gap_factor = a * (S_m + S) + a * a * K * B0 / (S_m * S)
    scale = abs(2.0 * muhat * max(B_mirror, B_m))
    upar_sq = 2.0 * muhat * w_gap * q_gap_factor / (B_m + B)
    upar_sq = _clip_upar_sq(upar_sq, scale, atol)
    numerator = Gamma * B * zm * cos_theta / S

    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / np.sqrt(upar_sq)

    endpoint_limit = _endpoint_theta_limit(zm, K, muhat, Gamma, a, B0)
    endpoint_gap = 0.5 * np.pi - np.abs(theta)
    if np.ndim(values) == 0:
        if endpoint_gap < 1.0e-8 and np.isfinite(endpoint_limit):
            return float(endpoint_limit)
        if not np.isfinite(values):
            return float(endpoint_limit)
        return float(values)

    near_endpoint = endpoint_gap < 1.0e-8
    if np.any(near_endpoint) and np.isfinite(endpoint_limit):
        values[near_endpoint] = endpoint_limit
    nonfinite = ~np.isfinite(values)
    if values.size and nonfinite[0]:
        values[0] = endpoint_limit
        nonfinite[0] = False
    if values.size > 1 and nonfinite[-1]:
        values[-1] = endpoint_limit
        nonfinite[-1] = False
    if np.any(nonfinite):
        finite_idx = np.flatnonzero(np.isfinite(values))
        if finite_idx.size == 0:
            raise ValueError("Could not evaluate the mirror time integrand.")
        bad_idx = np.flatnonzero(nonfinite)
        right_pos = np.searchsorted(finite_idx, bad_idx, side="left")
        left_pos = np.clip(right_pos - 1, 0, finite_idx.size - 1)
        right_pos = np.clip(right_pos, 0, finite_idx.size - 1)
        left_nearest = finite_idx[left_pos]
        right_nearest = finite_idx[right_pos]
        use_left = np.abs(bad_idx - left_nearest) <= np.abs(bad_idx - right_nearest)
        nearest = np.where(use_left, left_nearest, right_nearest)
        values[bad_idx] = values[nearest]
    return values


def _cumulative_theta_integral(theta, values):
    if hasattr(integrate, "cumulative_simpson") and theta.size >= 3:
        cumulative = integrate.cumulative_simpson(values, x=theta, initial=0.0)
        if np.all(np.diff(cumulative) > 0.0):
            return cumulative
    return integrate.cumulative_trapezoid(values, theta, initial=0.0)


def _make_inverse_interpolator(x, y):
    keep = np.concatenate(([True], np.diff(x) > 0.0))
    x_strict = x[keep]
    y_strict = y[keep]
    if x_strict.size < 2:
        raise ValueError("The time grid is not strictly increasing.")
    return interpolate.PchipInterpolator(x_strict, y_strict, extrapolate=False)


def _bounced_z_grid(t_array, z0, sigma0, zm, K, B_mirror, muhat, Gamma, a, B0, n_grid, atol):
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_grid)
    values = _theta_integrand(theta, zm, K, B_mirror, muhat, Gamma, a, B0, atol)
    I_grid = _cumulative_theta_integral(theta, values)
    T_half = float(I_grid[-1])
    if not np.isfinite(T_half) or T_half <= 0.0:
        raise ValueError("Invalid half-bounce time computed from the grid.")

    I_of_theta = interpolate.PchipInterpolator(theta, I_grid, extrapolate=False)
    theta_of_I = _make_inverse_interpolator(I_grid, theta)
    theta0 = np.arcsin(np.clip(z0 / zm, -1.0, 1.0))
    chi0 = float(I_of_theta(theta0))

    period = 2.0 * T_half
    chi = np.mod(chi0 + sigma0 * t_array, period)
    target = np.where(chi <= T_half, chi, period - chi)
    target = np.clip(target, 0.0, T_half)
    theta_t = theta_of_I(target)
    return zm * np.sin(theta_t), T_half


def _quad_theta_integrand_scalar(theta, zm, K, B_mirror, muhat, Gamma, a, B0, atol):
    return _theta_integrand(theta, zm, K, B_mirror, muhat, Gamma, a, B0, atol)


def _quad_I(theta, zm, K, B_mirror, muhat, Gamma, a, B0, rtol, atol):
    theta_min = -0.5 * np.pi
    if theta <= theta_min:
        return 0.0
    value, _ = integrate.quad(
        _quad_theta_integrand_scalar,
        theta_min,
        theta,
        args=(zm, K, B_mirror, muhat, Gamma, a, B0, atol),
        epsabs=atol,
        epsrel=rtol,
        limit=300,
    )
    return float(value)


def _bounced_z_quad(t_array, z0, sigma0, zm, K, B_mirror, muhat, Gamma, a, B0, n_grid, rtol, atol):
    theta_min = -0.5 * np.pi
    theta_max = 0.5 * np.pi
    theta0 = np.arcsin(np.clip(z0 / zm, -1.0, 1.0))

    T_half = _quad_I(theta_max, zm, K, B_mirror, muhat, Gamma, a, B0, rtol, atol)
    chi0 = _quad_I(theta0, zm, K, B_mirror, muhat, Gamma, a, B0, rtol, atol)
    if not np.isfinite(T_half) or T_half <= 0.0:
        raise ValueError("Invalid half-bounce time computed by quadrature.")

    n_bracket = max(32, min(int(n_grid), 20000))
    theta_grid = np.linspace(theta_min, theta_max, n_bracket)
    values = _theta_integrand(theta_grid, zm, K, B_mirror, muhat, Gamma, a, B0, atol)
    I_grid = _cumulative_theta_integral(theta_grid, values)
    period = 2.0 * T_half
    chi = np.mod(chi0 + sigma0 * t_array, period)
    targets = np.where(chi <= T_half, chi, period - chi)
    targets = np.clip(targets, 0.0, T_half)

    z_values = np.empty_like(targets, dtype=float)
    for i, target in enumerate(targets):
        if target <= atol:
            theta_i = theta_min
        elif T_half - target <= atol:
            theta_i = theta_max
        else:
            idx = int(np.searchsorted(I_grid, target, side="left"))
            lo_idx = max(0, idx - 2)
            hi_idx = min(theta_grid.size - 1, idx + 2)
            lo = theta_grid[lo_idx]
            hi = theta_grid[hi_idx]

            def residual(th):
                return _quad_I(th, zm, K, B_mirror, muhat, Gamma, a, B0, rtol, atol) - target

            if residual(lo) > 0.0 or residual(hi) < 0.0:
                lo, hi = theta_min, theta_max
            theta_i = optimize.brentq(
                residual,
                lo,
                hi,
                xtol=max(atol, 1.0e-14),
                rtol=max(rtol, 4.0 * np.finfo(float).eps),
            )
        z_values[i] = zm * np.sin(theta_i)
    return z_values, T_half


def _monotonic_no_mirror_z(t_array, z0, upar0, Gamma, K, a, B0, rtol, atol):
    if upar0 == 0.0:
        return np.full_like(t_array, z0, dtype=float)
    if K == 0.0:
        return z0 + (upar0 / Gamma) * t_array

    speed_sign = np.sign(upar0)
    abs_upar0 = abs(upar0)

    def time_integrand(z):
        return Gamma * _Bmag(z, K, a, B0) / (_S(z, a, B0) * abs_upar0)

    def signed_time_to_z(z):
        if z == z0:
            return 0.0
        lo, hi = (z0, z) if z > z0 else (z, z0)
        value, _ = integrate.quad(time_integrand, lo, hi, epsabs=atol, epsrel=rtol, limit=300)
        return value if z > z0 else -value

    z_values = np.empty_like(t_array, dtype=float)
    for i, t_value in enumerate(t_array):
        target = speed_sign * t_value
        if target == 0.0:
            z_values[i] = z0
            continue

        if target > 0.0:
            lo = z0
            hi = z0 + max(1.0, target * abs_upar0 / max(Gamma, 1.0))
            while signed_time_to_z(hi) < target:
                hi = z0 + 2.0 * (hi - z0) + 1.0
        else:
            hi = z0
            lo = z0 - max(1.0, -target * abs_upar0 / max(Gamma, 1.0))
            while signed_time_to_z(lo) > target:
                lo = z0 - 2.0 * (z0 - lo) - 1.0

        z_values[i] = optimize.brentq(
            lambda z: signed_time_to_z(z) - target,
            lo,
            hi,
            xtol=max(atol, 1.0e-14),
            rtol=max(rtol, 4.0 * np.finfo(float).eps),
        )
    return z_values


def theoretical_mirror_position(
    t,
    R0,
    u0,
    muhat=None,
    a=0.1,
    B0=1000.0,
    c=1.0,
    n_grid=100000,
    rtol=1e-12,
    atol=1e-14,
    method="grid",
    return_diagnostics=False,
    initial_direction=+1,
):
    """Return the theoretical no-drift guiding-center position R(t).

    The field is ``B=(-a*x*z, -a*y*z, B0 + a*z**2)`` with ``E=0``.  With
    perpendicular guiding-center drifts disabled, the guiding center stays on
    the magnetic field line ``r(z)**2 * (B0 + a*z**2) = K``.  The parallel
    motion is found from conservation of the relativistic magnetic moment per
    unit rest mass, ``muhat = mu_r/m0``, and the conserved Lorentz factor

        Gamma = sqrt(1 + (upar0**2 + 2*muhat*B_initial)/c**2).

    Parameters use any consistent normalized unit system.  ``u0`` is proper
    velocity, ``u = gamma*v``.  Times are in the corresponding time unit.

    Parameters
    ----------
    t : float or 1D array
        Time or times at which to evaluate the trajectory.
    R0, u0 : array-like, shape (3,)
        Initial guiding-center position and proper velocity.
    muhat : float, optional
        Relativistic magnetic moment per unit rest mass.  If omitted it is
        computed from the perpendicular proper velocity at ``R0``.
    a, B0, c : float
        Mirror-field parameters and speed of light.  ``a``, ``B0``, and ``c``
        must be positive.
    n_grid : int
        Resolution of the theta grid used by ``method="grid"`` and the
        bracketing grid used by ``method="quad"``.
    rtol, atol : float
        Relative and absolute tolerances for root finding and quadrature.
    method : {"grid", "quad"}
        ``"grid"`` builds and inverts a dense cumulative time grid in
        ``z = zm*sin(theta)``.  ``"quad"`` uses adaptive quadrature plus
        Brent inversion and is slower but useful for high-precision checks.
    return_diagnostics : bool
        If true, also return a diagnostics dictionary.
    initial_direction : {+1, -1}
        Direction used when ``upar0`` is exactly zero.  At a mirror point the
        reflected branch is selected automatically by the bounce map.

    Returns
    -------
    positions : ndarray
        Shape ``(len(t), 3)`` for array input, or shape ``(3,)`` for scalar
        input.
    diagnostics : dict, optional
        Returned when ``return_diagnostics=True``.
    """
    if a <= 0.0:
        raise ValueError(f"a must be positive; got {a}.")
    if B0 <= 0.0:
        raise ValueError(f"B0 must be positive; got {B0}.")
    if c <= 0.0:
        raise ValueError(f"c must be positive; got {c}.")
    if int(n_grid) < 8:
        raise ValueError(f"n_grid must be at least 8; got {n_grid}.")
    if initial_direction not in (-1, +1):
        raise ValueError("initial_direction must be +1 or -1.")
    if method not in ("grid", "quad"):
        raise ValueError('method must be "grid" or "quad".')

    t_array, scalar_input = _as_time_array(t)
    R0 = _as_vector3("R0", R0)
    u0 = _as_vector3("u0", u0)
    x0, y0, z0 = R0
    r0 = float(np.hypot(x0, y0))
    phi0 = float(np.arctan2(y0, x0))
    K = float(r0 * r0 * _S(z0, a, B0))
    B_initial = float(_Bmag(z0, K, a, B0))

    B_vec0 = _mirror_B_vector(R0, a, B0)
    b0 = B_vec0 / B_initial
    upar0 = float(np.dot(u0, b0))
    u0_sq = float(np.dot(u0, u0))
    uperp0_sq = u0_sq - upar0 * upar0
    uperp_tol = max(1000.0 * np.finfo(float).eps * max(u0_sq, upar0 * upar0, 1.0), 100.0 * atol)
    if uperp0_sq < -uperp_tol:
        raise ValueError(
            "Computed initial perpendicular proper-velocity squared is negative "
            f"({uperp0_sq:.6e})."
        )
    uperp0_sq = max(uperp0_sq, 0.0)

    if muhat is None:
        muhat_value = uperp0_sq / (2.0 * B_initial)
    else:
        muhat_value = float(muhat)
    if muhat_value < 0.0:
        raise ValueError(f"muhat must be non-negative; got {muhat_value}.")

    Gamma = float(np.sqrt(1.0 + (upar0 * upar0 + 2.0 * muhat_value * B_initial) / (c * c)))
    sigma0 = np.sign(upar0)
    if sigma0 == 0.0:
        sigma0 = float(initial_direction)

    z_minus = np.nan
    z_plus = np.nan
    T_half = np.inf
    T_bounce = np.inf
    B_mirror = np.inf

    if muhat_value == 0.0:
        z_t = _monotonic_no_mirror_z(t_array, z0, upar0, Gamma, K, a, B0, rtol, atol)
    else:
        zm, B_mirror = _find_positive_mirror_z(
            z0, B_initial, upar0, muhat_value, K, a, B0, rtol, atol
        )
        z_minus = -zm
        z_plus = zm

        if zm == 0.0:
            z_t = np.zeros_like(t_array, dtype=float)
            T_half = 0.0
            T_bounce = 0.0
        elif method == "grid":
            z_t, T_half = _bounced_z_grid(
                t_array,
                z0,
                sigma0,
                zm,
                K,
                B_mirror,
                muhat_value,
                Gamma,
                a,
                B0,
                int(n_grid),
                atol,
            )
            T_bounce = 2.0 * T_half
        else:
            z_t, T_half = _bounced_z_quad(
                t_array,
                z0,
                sigma0,
                zm,
                K,
                B_mirror,
                muhat_value,
                Gamma,
                a,
                B0,
                int(n_grid),
                rtol,
                atol,
            )
            T_bounce = 2.0 * T_half

    positions = _positions_from_z(z_t, K, phi0, a, B0)
    if scalar_input:
        positions = positions[0]

    diagnostics = {
        "upar0": upar0,
        "muhat": muhat_value,
        "Gamma": Gamma,
        "K": K,
        "phi0": phi0,
        "z_minus": z_minus,
        "z_plus": z_plus,
        "T_half": T_half,
        "T_bounce": T_bounce,
        "B_initial": B_initial,
        "B_mirror": B_mirror,
    }

    if return_diagnostics:
        return positions, diagnostics
    return positions


if __name__ == "__main__":
    t = np.linspace(0.0, 1000.0, 1000)

    R0 = np.array([1.0, 0.0, 0.0])
    u0 = np.array([0.02, 0.02, 0.01])
    pos, diag = theoretical_mirror_position(t, R0, u0, return_diagnostics=True)
    print(diag)
    print(pos[:3])

    R0 = np.array([0.0, 0.0, 1.0])
    u0 = np.array([0.02, 0.0, 0.01])
    pos = theoretical_mirror_position(t, R0, u0)
    print(pos[:3])
