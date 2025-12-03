#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math as m
from astropy.constants import M_sun
import os
from compact_object import NS
from accretion import (
    magnetic_torque_dai,
    magnetic_torque_dai_propeller,
    accretion_torque_dai,
    magnetospheric_radius,
    magnetic_torque_wang,
    accretion_torque,
    fastness_parameter,
    magnetospheric_radius_wang,
)

home = os.getenv("HOME")

plt.style.use("%s/.config/matplotlib/stylelib/presentation.mplstyle" % home)

Bs = np.geomspace(10**8, 10**14, 500)
M_NS = 1.4 * M_sun
NS = NS(P_NS=15, Pso=0, M_NS=M_NS, chi=0, alpha=0)
mdot = 0.01
Mdot = mdot * NS.Medd
psi = 1

# Wang approach
Rmags = np.array([magnetospheric_radius_wang(Mdot, B, NS, psi)[0] for B in Bs])
fastness = fastness_parameter(Rmags, NS.Rco)
Nwang = magnetic_torque_wang(Mdot, Rmags, NS.Rco, NS.M) + accretion_torque(
    Mdot, Rmags, NS.M
)
norm = accretion_torque(Mdot, Rmags, NS.M)

torque_fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.1})
axes[0].plot(
    fastness[fastness < 1],
    (Nwang / norm)[fastness < 1],
    label="Wang",
    ls="--",
    color="black",
)
norm = (Bs * NS.R_NS**3) ** 2 / (3 * NS.Rco**3)
axes[1].plot(
    fastness[fastness < 1], (Nwang / norm)[fastness < 1], ls="--", color="black"
)

# Dai approach
Rmags = magnetospheric_radius(Mdot, Bs, NS, psi)
fastness = fastness_parameter(Rmags, NS.Rco)
Ndai = np.where(
    fastness <= 1,
    magnetic_torque_dai(Bs, Rmags, NS),
    magnetic_torque_dai_propeller(Bs, Rmags, NS),
) + accretion_torque_dai(Mdot, Rmags, NS, gamma=0.9)
norm = accretion_torque(Mdot, Rmags, NS.M)
axes[0].plot(fastness, Ndai / norm, label="Dai", ls="solid", color="black")
norm = (Bs * NS.R_NS**3) ** 2 / (3 * NS.Rco**3)
axes[1].plot(fastness, Ndai / norm, ls="solid", color="black")


axes[0].set_ylabel(r"Dimensionless $N$")
for ax in axes:
    ax.set_ylim(-1.1, 2)
    ax.set_xlabel("$\omega$")
    ax.set_xlim(0, 3)
    ax.axhline(0, ls=":", color="black")
    ax.minorticks_on()
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))
axes[0].legend()
torque_fig.savefig("fig1_dai_2006.png", bbox_inches="tight", dpi=300)
