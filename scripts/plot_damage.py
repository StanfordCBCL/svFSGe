#!/usr/bin/env python
# coding=utf-8

import pdb
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1, 11, 11)
z = np.linspace(0, 15.0, 60)

T, Z = np.meshgrid(t, z)

mu0 = 89.71
KsKi0 = 0.35

lo = 15.0
z_om = lo / 2.0
z_od = lo / 4.0
vz = 2
phi_e_hm = 0.65

endtime = np.max(t)

f_time = 1.0 - pow((T - 1.0) / (endtime - 1.0), 1.0 / 1.5)
f_axi = np.exp(-pow(abs((Z - z_om) / z_od), vz))

res = {}
res["mu"] = mu0 * (f_time + (1.0 - f_time) * (1.0 - phi_e_hm * f_axi))
res["KsKi"] = KsKi0 * (f_time + (1.0 - f_time) * (1.0 - f_axi))

fig, ax = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={"projection": "3d"})
for i, (k, v) in enumerate(res.items()):
    ax[i].plot_surface(T, Z, v)
    ax[i].set_xlabel("t")
    ax[i].set_ylabel("z")
    ax[i].set_zlabel(k)

plt.show()
