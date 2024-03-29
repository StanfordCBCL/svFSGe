#!/usr/bin/env python
# coding=utf-8

#  ------------  marcos.latorre@yale.edu (2017)  ------------

import pdb
import os
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


def LaplaceMean(ro, lz, input_):
    # **	it computes the distensional Pressure (P) from the Laplace
    # 	equilibrium equation for given:
    # 	- current outer radius (ro)
    # 	- axial stretch from tf configuration (lz)
    # 	- mean mechanical properties and tf geometry (input)

    # ** retrieve parameters in 'input'
    c, c1t, c2t, c1d, c2d, alp, ritf, rotf = input_

    htf = rotf - ritf

    # ** current geometry is known from incompressibility
    ri = np.sqrt(ro**2 + 1 / lz * (ritf**2 - rotf**2))

    h = ro - ri

    # ** stretches
    lt = (2 * ri + h) / (2 * ritf + htf)
    lr = 1 / lt / lz
    ld = np.sqrt(lt**2 * np.sin(alp) ** 2 + lz**2 * np.cos(alp) ** 2)

    # ** circumferential Cauchy stress
    st = (
        c * (lt**2 - lr**2)
        + c1t * (lt**2 - 1) * np.exp(c2t * (lt**2 - 1) ** 2) * lt**2
        + 2
        * c1d
        * (ld**2 - 1)
        * np.exp(c2d * (ld**2 - 1) ** 2)
        * lt**2
        * np.sin(alp) ** 2
    )

    # ** Pressure from Laplace equilibrium equation
    P = st * h / ri

    return P


def MeandL(par, Pl, input_):
    # **	it generates diameter-Force data from a Pd test for given:
    # 	- mean mechanical properties (par)
    # 	- pairs of Pressure-stretch measurements (Pl)
    # 	- geometry at traction-free configuration (input)

    # ** par: material parameters for arterial MEAN response
    c, c1t, c2t, c1z, c2z, c1d, c2d, alp = par

    # ** Pl: Pressure and axial stretch during Pd test
    P, lz = Pl

    # ** input: some auxiliary parameters
    ritf, rotf = input_

    htf = rotf - ritf

    # ** current geometry is unknown at each deformation state
    #   compute outer diameter from Laplace equilibrium equation
    ro = np.ones(len(P))
    exitflag = np.ones(len(P))
    funvalue = np.ones(len(P))

    r0 = rotf

    inputLap = np.array([c, c1t, c2t, c1d, c2d, alp, ritf, rotf])

    xtol = 1e-12

    for i in np.arange(len(P)):
        if i > 1:
            r0 = ro[i - 1]
        # * solve non-linear equilibrium equation for given P and lz -> ro
        ro[i], info, eF, msg = scipy.optimize.fsolve(
            lambda r: P[i] - LaplaceMean(r, lz[i], inputLap),
            r0,
            xtol=xtol,
            full_output=True,
        )
        exitflag[i] = eF
        funvalue[i] = info["fvec"]

        if info["fvec"] > xtol or eF != 1:
            ro[i] = np.nan
            print(msg)

    # ** current geometry is known at each deformation state from incompressibility
    ri = np.sqrt(ro**2 + 1.0 / lz * (ritf**2 - rotf**2))
    h = ro - ri
    S = np.pi / lz * (rotf**2 - ritf**2)

    # ** stretches
    lt = (2 * ri + h) / (2 * ritf + htf)
    lr = 1.0 / lt / lz
    ld = np.sqrt(lt**2 * np.sin(alp) ** 2 + lz**2 * np.cos(alp) ** 2)

    # ** axial Cauchy stress
    sz = (
        c * (lz**2 - lr**2)
        + np.multiply(
            np.multiply(c1z * (lz**2 - 1), np.exp(c2z * (lz**2 - 1) ** 2)), lz**2
        )
        + np.multiply(
            np.multiply(2 * c1d * (ld**2 - 1), np.exp(c2d * (ld**2 - 1) ** 2)),
            lz**2,
        )
        * np.cos(alp) ** 2
    )

    # ** outer diameter and transducer Force during the Pd test
    dL = np.zeros((len(P), 2))
    dL[:, 0] = 2 * ro
    dL[:, 1] = np.multiply(sz, S) - np.pi * ri**2.0 * P

    return dL


def run_forward():
    mmHg_to_kPa = 0.133322

    # Po = np.array([104.2 * mmHg_to_kPa])
    Po = np.array(
        [104.9 * mmHg_to_kPa]
    )  # Original homeostatic pressure to get 0 displacement [kPa]

    output = get_geometry(Po)
    for k, v in output.items():
        print(k, "\t\t", v)


def plot():
    mmhg_to_kpa = 0.133322
    po = np.linspace(0, 200, 200) * mmhg_to_kpa

    rio = []
    for p in po:
        rio += [get_geometry([p])["rio"][0]]

    fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
    ax.plot(po / mmhg_to_kpa, rio)
    ax.grid(True)
    ax.set_xlabel("Po [mmHg]")
    ax.set_ylabel("rio [mm]")
    plt.show()


def run_inverse():
    mmHg_to_kPa = 0.133322

    rio_target = 1.6

    Po_cmame = 104.9 * mmHg_to_kPa
    Po_ini = 13.332

    m = lambda Po: rio_target - get_geometry([Po])["rio"][0]

    # ro[i], info, eF, msg = scipy.optimize.fsolve(lambda r: P[i] - LaplaceMean(r, lz[i], inputLap), r0, full_output=True)
    pdb.set_trace()
    Po, info, flag, msg = scipy.optimize.fsolve(
        lambda Po: rio_target - get_geometry([Po])["rio"][0], Po_ini, full_output=True
    )

    output = get_geometry(Po)
    for k, v in output.items():
        print(k, "\t\t", v)


def get_geometry(Po):
    # rotf0 = 0.444
    rotf0 = 0.4

    htf = 0.112

    ritf = rotf0 - htf

    lzivo = np.array([1.62])

    meanpar0 = np.array([18.536, 16.593, 0.108, 25.37, 0.036, 0.078, 1.719, 0.5024])

    dLPd = MeandL(meanpar0, np.array([Po, lzivo]), np.array([ritf, rotf0]))

    roo = dLPd[0] / 2

    rio = np.sqrt(roo**2 + 1 / lzivo * (ritf**2 - rotf0**2))

    ho = roo - rio

    # GenCylMesh
    output = {"Po": Po, "rio": rio, "roo": roo, "ho": ho}
    return output


if __name__ == "__main__":
    # rio = 0.6468;	   # 0.6468 | 0.5678 | 0.3984    Inner radius [mm]
    # ho  = 0.0402;	   # 0.0402 | 0.0343 | 0.0288    Thickness [mm]

    plot()
    # run_forward()
    # run_inverse()
