"""High-precision one-step full-orbit reference in the curved B field.

The magnetic field is

    B(x, y, z) = B0 * (-y, x, 0) / sqrt(x**2 + y**2) = B0 e_phi.

This module uses the exact full Lorentz equations rewritten in cylindrical
coordinates. It does not use guiding-center equations. The primary reference
integrates the reduced four-variable system built from the invariants

    ell = r u_phi
    k = u_z - Omega0 r

where Omega0 = q_over_m * B0 / c. With the default electron convention
q_over_m = -1 and B0 > 0, Omega0 is negative.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


R_MIN = 1.0e-14


def _as_vector3(name, value):
    """Return *value* as a float vector with shape (3,)."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector; got shape {vector.shape}.")
    return vector


def cart_to_cyl(x, u, *, r_min=R_MIN):
    """Convert Cartesian position and proper velocity to cylindrical components."""
    x = _as_vector3("x", x)
    u = _as_vector3("u", u)

    x_cart, y_cart, z = x
    ux, uy, uz = u
    r = np.hypot(x_cart, y_cart)
    if r <= r_min:
        raise ValueError(
            f"Cannot convert to cylindrical coordinates at r={r:.3e}; "
            f"the curved magnetic field is singular near the axis."
        )

    phi = np.arctan2(y_cart, x_cart)
    u_r = (x_cart * ux + y_cart * uy) / r
    u_phi = (-y_cart * ux + x_cart * uy) / r
    u_z = uz

    return r, phi, z, u_r, u_phi, u_z


def cyl_to_cart(r, phi, z, u_r, u_phi, u_z):
    """Convert cylindrical position and proper-velocity components to Cartesian."""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x = np.array([r * cos_phi, r * sin_phi, z], dtype=float)
    u = np.array(
        [
            u_r * cos_phi - u_phi * sin_phi,
            u_r * sin_phi + u_phi * cos_phi,
            u_z,
        ],
        dtype=float,
    )

    return x, u


def gamma_from_u(u, c=1.0):
    """Relativistic gamma from proper velocity u = gamma v."""
    if c <= 0.0:
        raise ValueError(f"c must be positive; got {c}.")
    u = np.asarray(u, dtype=float)
    return np.sqrt(1.0 + np.sum(u * u, axis=-1) / (c * c))


def curved_phi_B(x, B0=1000.0, *, r_min=R_MIN):
    """Magnetic field B = B0 e_phi in Cartesian components."""
    x = _as_vector3("x", x)
    x_cart, y_cart, _ = x
    r = np.hypot(x_cart, y_cart)
    if r <= r_min:
        raise ValueError(
            f"Cannot evaluate B at r={r:.3e}; the field direction is singular near the axis."
        )
    return B0 * np.array([-y_cart / r, x_cart / r, 0.0], dtype=float)


def _reduced_cylindrical_rhs(t, y, *, gamma0, omega0, ell, k, r_min):
    """RHS for [r, phi, z, u_r] using the exact cylindrical invariants."""
    del t
    r, phi, z, u_r = y
    del phi, z

    if r <= r_min:
        raise ValueError(
            f"Reduced cylindrical orbit reached r={r:.3e}; cannot continue through axis."
        )

    u_z = k + omega0 * r

    return np.array(
        [
            u_r / gamma0,
            ell / (gamma0 * r * r),
            u_z / gamma0,
            (ell * ell / (r * r * r) - omega0 * u_z) / gamma0,
        ],
        dtype=float,
    )


def reference_step_cylindrical(
    x0,
    u0,
    dt,
    B0=1000.0,
    q_over_m=-1.0,
    c=1.0,
    rtol=1e-13,
    atol=1e-15,
):
    """Advance one full-orbit step using the reduced cylindrical equations.

    The integration state is [r, phi, z, u_r]. The azimuthal and axial proper
    velocity components are reconstructed from ell = r u_phi and
    k = u_z - Omega0 r after the solve.
    """
    x0 = _as_vector3("x0", x0)
    u0 = _as_vector3("u0", u0)
    dt = float(dt)

    r0, phi0, z0, u_r0, u_phi0, u_z0 = cart_to_cyl(x0, u0)
    gamma0 = float(gamma_from_u(u0, c=c))
    omega0 = float(q_over_m * B0 / c)
    ell0 = float(r0 * u_phi0)
    k0 = float(u_z0 - omega0 * r0)

    if dt == 0.0:
        diag = {
            "gamma_initial": gamma0,
            "gamma_final": gamma0,
            "gamma_error": 0.0,
            "ell_initial": ell0,
            "ell_final": ell0,
            "ell_error": 0.0,
            "k_initial": k0,
            "k_final": k0,
            "k_error": 0.0,
            "nfev": 0,
            "success": True,
            "message": "dt is zero; no integration performed.",
        }
        return x0.copy(), u0.copy(), diag

    y0 = np.array([r0, phi0, z0, u_r0], dtype=float)
    def rhs(t, y):
        return _reduced_cylindrical_rhs(
            t,
            y,
            gamma0=gamma0,
            omega0=omega0,
            ell=ell0,
            k=k0,
            r_min=R_MIN,
        )

    sol = solve_ivp(
        rhs,
        (0.0, dt),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
        vectorized=False,
    )

    r1, phi1, z1, u_r1 = sol.y[:, -1]
    u_phi1 = ell0 / r1
    u_z1 = k0 + omega0 * r1
    x_ref, u_ref = cyl_to_cart(r1, phi1, z1, u_r1, u_phi1, u_z1)

    gamma1 = float(gamma_from_u(u_ref, c=c))
    ell1 = float(r1 * u_phi1)
    k1 = float(u_z1 - omega0 * r1)

    diag = {
        "gamma_initial": gamma0,
        "gamma_final": gamma1,
        "gamma_error": abs(gamma1 - gamma0),
        "ell_initial": ell0,
        "ell_final": ell1,
        "ell_error": abs(ell1 - ell0),
        "k_initial": k0,
        "k_final": k1,
        "k_error": abs(k1 - k0),
        "nfev": sol.nfev,
        "success": sol.success,
        "message": sol.message,
    }

    return x_ref, u_ref, diag


def _cart_to_cyl_many(x, u, *, r_min=R_MIN):
    """Vectorized Cartesian-to-cylindrical conversion for arrays of particles."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must have shape (n, 3); got {x.shape}.")
    if u.shape != x.shape:
        raise ValueError(f"u must have shape {x.shape}; got {u.shape}.")

    x_cart = x[:, 0]
    y_cart = x[:, 1]
    r = np.hypot(x_cart, y_cart)
    if np.any(r <= r_min):
        raise ValueError(
            f"At least one particle has r <= {r_min:.3e}; "
            "the curved magnetic field is singular near the axis."
        )

    phi = np.arctan2(y_cart, x_cart)
    u_r = (x_cart * u[:, 0] + y_cart * u[:, 1]) / r
    u_phi = (-y_cart * u[:, 0] + x_cart * u[:, 1]) / r
    u_z = u[:, 2]

    return r, phi, x[:, 2], u_r, u_phi, u_z


def _cyl_to_cart_many(r, phi, z, u_r, u_phi, u_z):
    """Vectorized cylindrical-to-Cartesian conversion for arrays of particles."""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x = np.column_stack((r * cos_phi, r * sin_phi, z))
    u = np.column_stack(
        (
            u_r * cos_phi - u_phi * sin_phi,
            u_r * sin_phi + u_phi * cos_phi,
            u_z,
        )
    )

    return x, u


def reference_step_cylindrical_many(
    x0,
    u0,
    dt,
    B0=1000.0,
    q_over_m=-1.0,
    c=1.0,
    rtol=1e-13,
    atol=1e-15,
    chunk_size=128,
):
    """Advance many particles with the reduced cylindrical full-orbit reference.

    This is the same reference system as reference_step_cylindrical(), integrated
    in batches so notebook comparisons over track files do not need one Python
    solve_ivp call per particle.
    """
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError(f"x0 must have shape (n, 3); got {x0.shape}.")
    if u0.shape != x0.shape:
        raise ValueError(f"u0 must have shape {x0.shape}; got {u0.shape}.")
    if chunk_size is None:
        chunk_size = len(x0)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}.")

    dt = float(dt)
    omega0 = float(q_over_m * B0 / c)
    x_ref = np.empty_like(x0, dtype=float)
    u_ref = np.empty_like(u0, dtype=float)
    diagnostics = {
        "gamma_error_max": 0.0,
        "ell_error_max": 0.0,
        "k_error_max": 0.0,
        "nfev": 0,
        "success": True,
        "message": "All chunks reached the end of the integration interval.",
        "chunks": [],
    }

    for start in range(0, len(x0), chunk_size):
        stop = min(start + chunk_size, len(x0))
        xs = x0[start:stop]
        us = u0[start:stop]

        r0, phi0, z0, u_r0, u_phi0, u_z0 = _cart_to_cyl_many(xs, us)
        gamma0 = gamma_from_u(us, c=c)
        ell0 = r0 * u_phi0
        k0 = u_z0 - omega0 * r0

        if dt == 0.0:
            r1, phi1, z1, u_r1 = r0, phi0, z0, u_r0
            nfev = 0
            success = True
            message = "dt is zero; no integration performed."
        else:
            n_chunk = stop - start
            y0 = np.vstack((r0, phi0, z0, u_r0)).ravel()

            def rhs(t, y):
                del t
                yy = y.reshape(4, n_chunk)
                r = yy[0]
                if np.any(r <= R_MIN):
                    raise ValueError(
                        "A reduced cylindrical orbit reached the singular axis."
                    )
                u_z = k0 + omega0 * r
                dy = np.empty_like(yy)
                dy[0] = yy[3] / gamma0
                dy[1] = ell0 / (gamma0 * r * r)
                dy[2] = u_z / gamma0
                dy[3] = (ell0 * ell0 / (r * r * r) - omega0 * u_z) / gamma0
                return dy.ravel()

            sol = solve_ivp(
                rhs,
                (0.0, dt),
                y0,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                dense_output=False,
                vectorized=False,
            )
            yf = sol.y[:, -1].reshape(4, n_chunk)
            r1, phi1, z1, u_r1 = yf
            nfev = sol.nfev
            success = sol.success
            message = sol.message

        u_phi1 = ell0 / r1
        u_z1 = k0 + omega0 * r1
        x_chunk, u_chunk = _cyl_to_cart_many(r1, phi1, z1, u_r1, u_phi1, u_z1)
        x_ref[start:stop] = x_chunk
        u_ref[start:stop] = u_chunk

        gamma1 = gamma_from_u(u_chunk, c=c)
        ell1 = r1 * u_phi1
        k1 = u_z1 - omega0 * r1

        gamma_error = float(np.max(np.abs(gamma1 - gamma0)))
        ell_error = float(np.max(np.abs(ell1 - ell0)))
        k_error = float(np.max(np.abs(k1 - k0)))
        diagnostics["gamma_error_max"] = max(
            diagnostics["gamma_error_max"], gamma_error
        )
        diagnostics["ell_error_max"] = max(diagnostics["ell_error_max"], ell_error)
        diagnostics["k_error_max"] = max(diagnostics["k_error_max"], k_error)
        diagnostics["nfev"] += int(nfev)
        diagnostics["success"] = bool(diagnostics["success"] and success)
        diagnostics["chunks"].append(
            {
                "start": start,
                "stop": stop,
                "nfev": int(nfev),
                "success": bool(success),
                "message": message,
                "gamma_error_max": gamma_error,
                "ell_error_max": ell_error,
                "k_error_max": k_error,
            }
        )

        if not success:
            diagnostics["message"] = message
            break

    return x_ref, u_ref, diagnostics


def direct_cartesian_rhs(t, y, B0=1000.0, q_over_m=-1.0, c=1.0):
    """Cartesian full Lorentz RHS for optional cross-checks only.

    The main reference in this module is reference_step_cylindrical(), not this
    direct Cartesian system.
    """
    del t
    state = np.asarray(y, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"y must be a length-6 state [x, u]; got shape {state.shape}.")

    x = state[:3]
    u = state[3:]
    gamma = float(gamma_from_u(u, c=c))
    dxdt = u / gamma
    B = curved_phi_B(x, B0=B0)
    dudt = (q_over_m / c) * np.cross(u / gamma, B)

    return np.concatenate([dxdt, dudt])


def reference_step_cartesian_cross_check(
    x0,
    u0,
    dt,
    B0=1000.0,
    q_over_m=-1.0,
    c=1.0,
    rtol=1e-13,
    atol=1e-15,
):
    """Integrate the Cartesian equations for an explicit non-reference check."""
    x0 = _as_vector3("x0", x0)
    u0 = _as_vector3("u0", u0)
    y0 = np.concatenate([x0, u0])

    sol = solve_ivp(
        direct_cartesian_rhs,
        (0.0, float(dt)),
        y0,
        args=(B0, q_over_m, c),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
        vectorized=False,
    )

    y1 = sol.y[:, -1]
    diag = {
        "gamma_initial": float(gamma_from_u(u0, c=c)),
        "gamma_final": float(gamma_from_u(y1[3:], c=c)),
        "nfev": sol.nfev,
        "success": sol.success,
        "message": sol.message,
    }
    diag["gamma_error"] = abs(diag["gamma_final"] - diag["gamma_initial"])

    return y1[:3], y1[3:], diag


def boris_step_B_only(x0, u_minus_half, dt, B_func, q_over_m=-1.0, c=1.0):
    """One relativistic Boris step for E = 0 in Cartesian coordinates.

    Boris proper velocities are staggered in time. The input u_minus_half and
    output u_plus are half-step velocities, while x_new is advanced over dt.
    Compare with the continuous reference only after aligning this timing
    convention with the reference initial data.
    """
    x0 = _as_vector3("x0", x0)
    u_minus = _as_vector3("u_minus_half", u_minus_half)
    dt = float(dt)

    gamma_minus = float(gamma_from_u(u_minus, c=c))
    B = _as_vector3("B_func(x0)", B_func(x0))

    t = (q_over_m * dt / (2.0 * c * gamma_minus)) * B
    s = 2.0 * t / (1.0 + np.dot(t, t))
    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)

    gamma_plus = float(gamma_from_u(u_plus, c=c))
    x_new = x0 + dt * u_plus / gamma_plus

    return x_new, u_plus


def run_constant_radius_test(dt=1e-4, B0=1000.0, q_over_m=-1.0, c=1.0):
    """Run and print an exact constant-radius orbit check."""
    omega0 = q_over_m * B0 / c
    if omega0 == 0.0:
        raise ValueError("Constant-radius test requires nonzero Omega0.")

    r0 = 1.0
    phi0 = 0.0
    z0 = 0.0
    u_r0 = 0.0
    u_phi0 = 0.1
    u_z0 = u_phi0 * u_phi0 / (omega0 * r0)

    x0, u0 = cyl_to_cart(r0, phi0, z0, u_r0, u_phi0, u_z0)
    x_ref, u_ref, diag = reference_step_cylindrical(
        x0, u0, dt, B0=B0, q_over_m=q_over_m, c=c
    )

    gamma0 = float(gamma_from_u(u0, c=c))
    phi_exact = phi0 + u_phi0 * dt / (gamma0 * r0)
    z_exact = z0 + u_z0 * dt / gamma0
    x_exact, u_exact = cyl_to_cart(
        r0, phi_exact, z_exact, u_r0, u_phi0, u_z0
    )

    abs_x_error = np.abs(x_ref - x_exact)
    abs_u_error = np.abs(u_ref - u_exact)
    max_abs_error = max(float(np.max(abs_x_error)), float(np.max(abs_u_error)))

    print("constant-radius |x_ref - x_exact| =", abs_x_error)
    print("constant-radius |u_ref - u_exact| =", abs_u_error)
    print("constant-radius max abs error =", max_abs_error)
    print("diagnostics =", diag)

    return {
        "x_ref": x_ref,
        "u_ref": u_ref,
        "x_exact": x_exact,
        "u_exact": u_exact,
        "abs_x_error": abs_x_error,
        "abs_u_error": abs_u_error,
        "max_abs_error": max_abs_error,
        "diagnostics": diag,
    }


if __name__ == "__main__":
    x0 = np.array([1.0, 0.0, 0.0])
    u0 = np.array([0.05, -0.05, 0.0001])
    dt = 1e-4

    x_ref, u_ref, diag = reference_step_cylindrical(x0, u0, dt)

    print(x_ref)
    print(u_ref)
    print(diag)

    run_constant_radius_test(dt=dt)
