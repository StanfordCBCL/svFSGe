#!/usr/bin/env python
# coding=utf-8

"""
Constrained mixture thin-wall model for arterial mechanics.

Implements Appendix E of Latorre & Humphrey (2020) CMAME, matching the
prestress case in gr_equilibrated.cpp. Computes geometry and stresses
for a thin-walled cylinder using G&R constitutive parameters.
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

MMHG_TO_KPA = 0.133322


@dataclass
class Params:
    """Material and geometric parameters (paper/C++ naming)."""
    # Geometry
    a_o: float   # inner radius [mm]
    h_o: float   # thickness [mm]
    l_o: float   # length [mm]
    # Mass fractions
    phi_e: float
    phi_m: float
    phi_c: float
    # Elastin (neo-Hookean)
    mu: float     # shear modulus [kPa]
    G_e_t: float  # circumferential deposition stretch
    G_e_z: float  # axial deposition stretch
    # Smooth muscle (Fung-type)
    c_m: float
    d_m: float
    G_m: float    # deposition stretch
    # Collagen (Fung-type)
    c_c: float
    d_c: float
    G_c: float    # deposition stretch
    # Collagen fractions (beta_t + beta_z + 2*beta_d = 1)
    beta_t: float
    beta_z: float
    beta_d: float
    # Fiber angle [rad]
    alpha: float
    # Active stress
    T_max: float
    lam_M: float
    lam_0: float


def load_params(xml_path, h_o=0.040):
    """Parse constitutive parameters from svFSI XML input file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cm = root.find('.//Constitutive_model')

    def get(tag):
        return float(cm.find(tag).text)

    return Params(
        a_o=get('rIo'),
        h_o=h_o,
        l_o=get('lo'),
        phi_e=get('phieo'),
        phi_m=get('phimo'),
        phi_c=get('phico'),
        mu=get('mu'),
        G_e_t=get('Get'),
        G_e_z=get('Gez'),
        c_m=get('cm'),
        d_m=get('dm'),
        G_m=get('Gm'),
        c_c=get('cc'),
        d_c=get('dc'),
        G_c=get('Gc'),
        beta_t=get('betat'),
        beta_z=get('betaz'),
        beta_d=get('betad'),
        alpha=get('alpha'),
        T_max=get('Tmax'),
        lam_M=get('lamM'),
        lam_0=get('lam0'),
    )


def _fung(c, d, lam2):
    """Fung-type fiber stress factor: c * (lam^2 - 1) * exp(d * (lam^2 - 1)^2)."""
    e = lam2 - 1.0
    return c * e * np.exp(d * e * e)


def cauchy_stress(lt, lz, par):
    """Compute Cauchy stress components (without Lagrange multiplier).

    Returns dict with keys 'rr', 'tt', 'zz' for radial, circumferential,
    and axial Cauchy stress contributions from all constituents.
    These are sigma_hat = (1/J) * F * S_material * F^T with J=1 (incompressible).
    """
    lr = 1.0 / (lt * lz)
    G_e_r = 1.0 / (par.G_e_t * par.G_e_z)

    # --- Elastin (neo-Hookean with deposition stretches) ---
    # Se = phi_e * mu * Ge^2 (2PK), push forward: sigma = lt^2 * Se_tt etc.
    se_rr = par.phi_e * par.mu * (G_e_r * lr) ** 2
    se_tt = par.phi_e * par.mu * (par.G_e_t * lt) ** 2
    se_zz = par.phi_e * par.mu * (par.G_e_z * lz) ** 2

    # --- Smooth muscle (Fung-type, circumferential only) ---
    lmt2 = (par.G_m * lt) ** 2
    sm_tt = par.phi_m * _fung(par.c_m, par.d_m, lmt2) * lmt2

    # --- Collagen (Fung-type, 4 families) ---
    lct2 = (par.G_c * lt) ** 2
    lcz2 = (par.G_c * lz) ** 2
    ld = np.sqrt(lt**2 * np.sin(par.alpha)**2 + lz**2 * np.cos(par.alpha)**2)
    lcd2 = (par.G_c * ld) ** 2

    Gc2 = par.G_c ** 2
    sin2a = np.sin(par.alpha) ** 2
    cos2a = np.cos(par.alpha) ** 2

    # Push-forward of dyad(Np) and dyad(Nn) to theta-theta: lt^2 * sin^2(alpha)
    # Push-forward of dyad(Np) and dyad(Nn) to zz: lz^2 * cos^2(alpha)
    sc_tt = par.phi_c * Gc2 * (
        par.beta_t * _fung(par.c_c, par.d_c, lct2) * lt**2
        + 2 * par.beta_d * _fung(par.c_c, par.d_c, lcd2) * lt**2 * sin2a
    )

    sc_zz = par.phi_c * Gc2 * (
        par.beta_z * _fung(par.c_c, par.d_c, lcz2) * lz**2
        + 2 * par.beta_d * _fung(par.c_c, par.d_c, lcd2) * lz**2 * cos2a
    )

    # --- Active stress (circumferential, carried by smooth muscle) ---
    # 2PK: Sa = Tmax * f1 * f2 * lt^2 * dyad(N1), push forward adds lt^2
    CB = np.sqrt(np.log(2.0))
    sa_tt = par.phi_m * par.T_max * (1.0 - np.exp(-CB**2)) * \
        (1.0 - ((par.lam_M - 1.0) / (par.lam_M - par.lam_0))**2) * lt**4

    return {
        'rr': se_rr,
        'tt': se_tt + sm_tt + sc_tt + sa_tt,
        'zz': se_zz + sc_zz,
    }


def cauchy_stress_theta(ro, lz, par):
    """Effective circumferential Cauchy stress (sigma_tt - sigma_rr) for thin wall."""
    ri, h, lt, _ = _geometry_from_ro(ro, lz, par)
    s = cauchy_stress(lt, lz, par)
    return s['tt'] - s['rr']


def _geometry_from_ro(ro, lz, par):
    """Compute current geometry from outer radius and axial stretch.

    Uses incompressibility to get inner radius, then midwall stretch.
    Returns (ri, h, lt, lr).
    """
    ro_ref = par.a_o + par.h_o
    ri = np.sqrt(ro**2 + (1.0 / lz) * (par.a_o**2 - ro_ref**2))
    h = ro - ri
    # Midwall circumferential stretch
    lt = (2 * ri + h) / (2 * par.a_o + par.h_o)
    lr = 1.0 / (lt * lz)
    return ri, h, lt, lr


def pressure_from_ro(ro, lz, par):
    """Compute pressure from Laplace equilibrium: P = sigma_tt * h / a."""
    ri, h, lt, _ = _geometry_from_ro(ro, lz, par)
    s = cauchy_stress(lt, lz, par)
    sigma_tt = s['tt'] - s['rr']
    return sigma_tt * h / ri


def solve_geometry(P, lz, par):
    """Solve for deformed geometry given pressure and axial stretch.

    Returns dict with geometric quantities.
    """
    ro_ref = par.a_o + par.h_o

    ro, info, flag, msg = scipy.optimize.fsolve(
        lambda r: P - pressure_from_ro(r, lz, par),
        ro_ref,
        full_output=True,
    )
    ro = float(ro[0])

    if flag != 1:
        raise RuntimeError(f"fsolve failed: {msg}")

    ri, h, lt, lr = _geometry_from_ro(ro, lz, par)
    s = cauchy_stress(lt, lz, par)

    return {
        'P_kPa': P,
        'P_mmHg': P / MMHG_TO_KPA,
        'a': ri,
        'ro': ro,
        'h': h,
        'lt': lt,
        'lz': lz,
        'lr': lr,
        'sigma_tt': s['tt'] - s['rr'],
        'sigma_zz': s['zz'] - s['rr'],
    }


def axial_force(P, ri, h, sigma_zz):
    """Reduced axial force (transducer force, closed-end condition)."""
    return sigma_zz * np.pi * h * (2 * ri + h) - P * np.pi * ri**2


def run_forward(par, pressure_mmhg=104.9):
    """Compute geometry at homeostatic pressure."""
    P = pressure_mmhg * MMHG_TO_KPA
    lz = 1.0

    result = solve_geometry(P, lz, par)
    fz = axial_force(P, result['a'], result['h'], result['sigma_zz'])
    result['F_z'] = fz

    print("=== Forward: geometry at homeostatic pressure ===")
    for k, v in result.items():
        print(f"  {k:12s} = {v:.6f}")
    return result


def vmax_to_Q(vmax, radius):
    """Flow rate from parabolic profile peak velocity: Q = pi * r^2 * vmax / 2.

    For a parabolic profile u(r) = vmax * (1 - r^2/R^2):
      Q = integral_0^R u(r) * 2*pi*r dr = pi * R^2 * vmax / 2

    Args:
        vmax: maximum (centerline) velocity [mm/s]
        radius: inner radius [mm]

    Returns:
        Flow rate [mm^3/s]
    """
    return np.pi * radius**2 * vmax / 2.0


def poiseuille_resistance(mu_blood, length, radius):
    """Poiseuille flow resistance: R = 8 * mu * L / (pi * r^4).

    Args:
        mu_blood: dynamic viscosity [kPa·s]
        length: vessel length [mm]
        radius: inner radius [mm]

    Returns:
        Resistance [kPa·s/mm^3]
    """
    return 8.0 * mu_blood * length / (np.pi * radius**4)


def run_inverse_pressure(par, pressure_mmhg):
    """Find equilibrium geometry for a given pressure.

    Args:
        par: material/geometric parameters
        pressure_mmhg: prescribed pressure [mmHg]
    """
    P = pressure_mmhg * MMHG_TO_KPA
    lz = 1.0

    result = solve_geometry(P, lz, par)
    fz = axial_force(P, result['a'], result['h'], result['sigma_zz'])
    result['F_z'] = fz

    print(f"=== Inverse: geometry at P = {pressure_mmhg:.2f} mmHg ===")
    for k, v in result.items():
        print(f"  {k:12s} = {v:.6f}")
    return result


def run_inverse_flow(par, delta_Q, mu_blood_Pa_s=0.004):
    """Find geometry in mechanobiological equilibrium with pulse pressure.

    Given delta_Q (systolic - diastolic flow rate), finds the vessel
    geometry where:
      1. Poiseuille resistance R = 8 * mu * L / (pi * a^4)
      2. Pulse pressure delta_P = R * delta_Q
      3. Thin-wall equilibrium: sigma_tt * h / a = delta_P

    Args:
        par: material/geometric parameters
        delta_Q: flow rate difference (systolic - diastolic) [mm^3/s]
        mu_blood_Pa_s: blood dynamic viscosity [Pa·s] (default: 0.004)
    """
    lz = 1.0
    mu_blood = mu_blood_Pa_s * 1e-3  # convert Pa·s to kPa·s

    def objective(ro):
        ri, h, lt, lr = _geometry_from_ro(ro, lz, par)
        # Poiseuille resistance
        R = poiseuille_resistance(mu_blood, par.l_o, ri)
        # Pulse pressure from flow
        delta_P = R * delta_Q
        # Equilibrium pressure from wall stress
        s = cauchy_stress(lt, lz, par)
        sigma_tt = s['tt'] - s['rr']
        P_wall = sigma_tt * h / ri
        return delta_P - P_wall

    ro_ref = par.a_o + par.h_o
    ro_sol, info, flag, msg = scipy.optimize.fsolve(
        objective, ro_ref, full_output=True
    )
    ro_sol = float(ro_sol[0])

    if flag != 1:
        raise RuntimeError(f"fsolve failed: {msg}")

    ri, h, lt, lr = _geometry_from_ro(ro_sol, lz, par)
    R = poiseuille_resistance(mu_blood, par.l_o, ri)
    delta_P = R * delta_Q
    result = solve_geometry(delta_P, lz, par)
    fz = axial_force(delta_P, result['a'], result['h'], result['sigma_zz'])
    result['F_z'] = fz
    result['delta_Q'] = delta_Q
    result['R'] = R
    result['mu_blood'] = mu_blood_Pa_s

    print("=== Inverse: geometry from flow rate difference ===")
    print(f"  {'delta_Q':12s} = {delta_Q:.6f} mm^3/s")
    print(f"  {'mu_blood':12s} = {mu_blood_Pa_s:.6f} Pa·s")
    print(f"  {'R':12s} = {R:.6e} kPa·s/mm^3")
    print(f"  {'delta_P':12s} = {delta_P:.6f} kPa ({delta_P / MMHG_TO_KPA:.2f} mmHg)")
    for k, v in result.items():
        if k not in ('delta_Q', 'R', 'mu_blood'):
            print(f"  {k:12s} = {v:.6f}")
    return result


def run_inverse_velocity(par, delta_vmax, mu_blood_Pa_s=0.004):
    """Find geometry in mechanobiological equilibrium given peak velocity difference.

    Given delta_vmax (systolic - diastolic peak velocity), finds the vessel
    geometry where:
      1. Q(ri) = pi * ri^2 * delta_vmax / 2  (parabolic profile, radius-dependent)
      2. Poiseuille resistance R = 8 * mu * L / (pi * ri^4)
      3. Pulse pressure delta_P = R * Q(ri)
      4. Thin-wall equilibrium: sigma_tt * h / ri = delta_P

    Args:
        par: material/geometric parameters
        delta_vmax: peak velocity difference (systolic - diastolic) [mm/s]
        mu_blood_Pa_s: blood dynamic viscosity [Pa·s] (default: 0.004)
    """
    lz = 1.0
    mu_blood = mu_blood_Pa_s * 1e-3  # convert Pa·s to kPa·s

    def objective(ro):
        ri, h, lt, lr = _geometry_from_ro(ro, lz, par)
        delta_Q = vmax_to_Q(delta_vmax, ri)
        R = poiseuille_resistance(mu_blood, par.l_o, ri)
        delta_P = R * delta_Q
        s = cauchy_stress(lt, lz, par)
        sigma_tt = s['tt'] - s['rr']
        P_wall = sigma_tt * h / ri
        return delta_P - P_wall

    ro_ref = par.a_o + par.h_o
    ro_sol, info, flag, msg = scipy.optimize.fsolve(
        objective, ro_ref, full_output=True
    )
    ro_sol = float(ro_sol[0])

    if flag != 1:
        raise RuntimeError(f"fsolve failed: {msg}")

    ri, h, lt, lr = _geometry_from_ro(ro_sol, lz, par)
    delta_Q = vmax_to_Q(delta_vmax, ri)
    R = poiseuille_resistance(mu_blood, par.l_o, ri)
    delta_P = R * delta_Q
    result = solve_geometry(delta_P, lz, par)
    fz = axial_force(delta_P, result['a'], result['h'], result['sigma_zz'])
    result['F_z'] = fz
    result['delta_vmax'] = delta_vmax
    result['delta_Q'] = delta_Q
    result['R'] = R
    result['mu_blood'] = mu_blood_Pa_s

    print("=== Inverse: geometry from peak velocity difference ===")
    print(f"  {'delta_vmax':12s} = {delta_vmax:.6f} mm/s")
    print(f"  {'delta_Q':12s} = {delta_Q:.6f} mm^3/s")
    print(f"  {'mu_blood':12s} = {mu_blood_Pa_s:.6f} Pa·s")
    print(f"  {'R':12s} = {R:.6e} kPa·s/mm^3")
    print(f"  {'delta_P':12s} = {delta_P:.6f} kPa ({delta_P / MMHG_TO_KPA:.2f} mmHg)")
    for k, v in result.items():
        if k not in ('delta_vmax', 'delta_Q', 'R', 'mu_blood'):
            print(f"  {k:12s} = {v:.6f}")
    return result


def run_plot_velocity(par, mu_blood_Pa_s=0.004):
    """Sweep delta_vmax and plot equilibrium inner radius vs peak velocity difference."""
    lz = 1.0
    mu_blood = mu_blood_Pa_s * 1e-3  # convert Pa·s to kPa·s
    delta_vmaxs = np.linspace(1, 5000, 200)
    radii = []
    pressures = []

    for dvmax in delta_vmaxs:
        def objective(ro):
            ri, h, lt, lr = _geometry_from_ro(ro, lz, par)
            delta_Q = vmax_to_Q(dvmax, ri)
            R = poiseuille_resistance(mu_blood, par.l_o, ri)
            delta_P = R * delta_Q
            s = cauchy_stress(lt, lz, par)
            sigma_tt = s['tt'] - s['rr']
            P_wall = sigma_tt * h / ri
            return delta_P - P_wall

        ro_ref = par.a_o + par.h_o
        try:
            ro_sol, info, flag, msg = scipy.optimize.fsolve(
                objective, ro_ref, full_output=True
            )
            if flag != 1:
                raise RuntimeError
            ro_sol = float(ro_sol[0])
            ri, h, lt, lr = _geometry_from_ro(ro_sol, lz, par)
            delta_Q = vmax_to_Q(dvmax, ri)
            R = poiseuille_resistance(mu_blood, par.l_o, ri)
            radii.append(ri)
            pressures.append(R * delta_Q / MMHG_TO_KPA)
        except RuntimeError:
            radii.append(np.nan)
            pressures.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    ax1.plot(delta_vmaxs, radii, 'b-', linewidth=2)
    ax1.set_xlabel(r'$\Delta v_\mathrm{max}$ [mm/s]')
    ax1.set_ylabel('Inner radius [mm]')
    ax1.set_title('Equilibrium radius vs peak velocity difference')
    ax1.grid(True)

    ax2.plot(delta_vmaxs, pressures, 'r-', linewidth=2)
    ax2.set_xlabel(r'$\Delta v_\mathrm{max}$ [mm/s]')
    ax2.set_ylabel('Pulse pressure [mmHg]')
    ax2.set_title('Pulse pressure vs peak velocity difference')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('initial_velocity.pdf')
    print("Saved plot to initial_velocity.pdf")


def run_plot_pressure(par):
    """Sweep pressure and plot inner radius vs pressure."""
    lz = 1.0
    pressures_mmhg = np.linspace(0.1, 200, 200)
    radii = []

    for p_mmhg in pressures_mmhg:
        P = p_mmhg * MMHG_TO_KPA
        try:
            result = solve_geometry(P, lz, par)
            radii.append(result['a'])
        except RuntimeError:
            radii.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(pressures_mmhg, radii, 'b-', linewidth=2)
    ax.set_xlabel('Pressure [mmHg]')
    ax.set_ylabel('Inner radius [mm]')
    ax.set_title('Thin-wall constrained mixture model')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('initial_pressure.pdf')
    print("Saved plot to initial_pressure.pdf")


def run_plot_flow(par, mu_blood_Pa_s=0.004):
    """Sweep delta_Q and plot equilibrium inner radius vs flow rate difference."""
    lz = 1.0
    mu_blood = mu_blood_Pa_s * 1e-3  # convert Pa·s to kPa·s
    delta_Qs = np.linspace(1, 5000, 200)
    radii = []
    pressures = []

    for dQ in delta_Qs:
        def objective(ro):
            ri, h, lt, lr = _geometry_from_ro(ro, lz, par)
            R = poiseuille_resistance(mu_blood, par.l_o, ri)
            delta_P = R * dQ
            s = cauchy_stress(lt, lz, par)
            sigma_tt = s['tt'] - s['rr']
            P_wall = sigma_tt * h / ri
            return delta_P - P_wall

        ro_ref = par.a_o + par.h_o
        try:
            ro_sol, info, flag, msg = scipy.optimize.fsolve(
                objective, ro_ref, full_output=True
            )
            if flag != 1:
                raise RuntimeError
            ro_sol = float(ro_sol[0])
            ri, h, lt, lr = _geometry_from_ro(ro_sol, lz, par)
            R = poiseuille_resistance(mu_blood, par.l_o, ri)
            radii.append(ri)
            pressures.append(R * dQ / MMHG_TO_KPA)
        except RuntimeError:
            radii.append(np.nan)
            pressures.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    ax1.plot(delta_Qs, radii, 'b-', linewidth=2)
    ax1.set_xlabel(r'$\Delta Q$ [mm$^3$/s]')
    ax1.set_ylabel('Inner radius [mm]')
    ax1.set_title('Equilibrium radius vs flow rate difference')
    ax1.grid(True)

    ax2.plot(delta_Qs, pressures, 'r-', linewidth=2)
    ax2.set_xlabel(r'$\Delta Q$ [mm$^3$/s]')
    ax2.set_ylabel('Pulse pressure [mmHg]')
    ax2.set_title('Pulse pressure vs flow rate difference')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('initial_flow.pdf')
    print("Saved plot to initial_flow.pdf")


def main():
    parser = argparse.ArgumentParser(
        description='Constrained mixture thin-wall model for arterial mechanics')
    parser.add_argument('--xml', required=True, help='Path to svFSI XML input file')
    parser.add_argument('--mode', choices=['forward', 'inverse', 'plot'],
                        default='forward')
    parser.add_argument('--pressure', type=float, default=104.9,
                        help='Pressure [mmHg] (forward/plot mode, or inverse with --pressure)')
    parser.add_argument('--delta-Q', type=float, default=None,
                        help='Systolic - diastolic flow rate [mm^3/s] (inverse/plot mode)')
    parser.add_argument('--delta-vmax', type=float, default=None,
                        help='Systolic - diastolic peak velocity [mm/s] (inverse/plot mode, parabolic profile)')
    parser.add_argument('--mu-blood', type=float, default=0.004,
                        help='Blood dynamic viscosity [Pa·s] (default: 0.004)')
    parser.add_argument('--h_o', type=float, default=0.040,
                        help='Reference wall thickness [mm] (default: Table 1 value)')
    args = parser.parse_args()

    par = load_params(args.xml, h_o=args.h_o)

    if args.mode == 'forward':
        run_forward(par, pressure_mmhg=args.pressure)
    elif args.mode == 'inverse':
        if args.delta_vmax is not None:
            run_inverse_velocity(par, delta_vmax=args.delta_vmax, mu_blood_Pa_s=args.mu_blood)
        elif args.delta_Q is not None:
            run_inverse_flow(par, delta_Q=args.delta_Q, mu_blood_Pa_s=args.mu_blood)
        else:
            run_inverse_pressure(par, pressure_mmhg=args.pressure)
    elif args.mode == 'plot':
        if args.delta_vmax is not None:
            run_plot_velocity(par, mu_blood_Pa_s=args.mu_blood)
        elif args.delta_Q is not None:
            run_plot_flow(par, mu_blood_Pa_s=args.mu_blood)
        else:
            run_plot_pressure(par)


if __name__ == '__main__':
    main()
