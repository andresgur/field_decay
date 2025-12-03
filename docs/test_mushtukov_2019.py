import numpy as np
from compact_object import NS
from accretion import mass_transfer_rate_mag_radius, magnetic_moment, spherization_radius_poutanen
import matplotlib.pyplot as plt

B = 10**12 
ns = NS(10)
mdots = np.arange(3, 10000)
Mdots = ns.Medd * mdots
ns = NS(10)
mu = magnetic_moment(B, ns.R_NS)
lss  = ["solid", "--", "-.", "dotted"]
e_winds = [0.5, 0.75 , 1]
fig = plt.figure()
for j, e_wind in enumerate(e_winds):
    mdotsin = []
    for Mdot in Mdots:
        Rsph = spherization_radius_poutanen(Mdot /ns.Medd, ns.Risco, e_wind)
        mass_transfered = mass_transfer_rate_mag_radius(Mdot, mu, ns.Medd, Rsph, ns.M, e_wind=e_wind)
        mdotsin.append(mass_transfered)
    plt.plot(mdots, mdotsin / (Mdots), label="$\epsilon_\omega$ = %.2f" % e_wind, ls=lss[j], color="black")

plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(direction="in", which="both")
plt.xlabel(r"$\dot{m}_\mathrm{0}$")
plt.ylabel(r"$\dot{M}_\mathrm{R_m}/\dot{M}_\mathrm{0}$")
plt.xscale("log")
plt.legend()
plt.ylim(0.35, 1.09)
plt.xlim(1, 10**3.9)
plt.savefig("mushtukov_2019_figure3.png")
plt.close(fig)

e_wind = 0.5
Bs = np.flip(np.geomspace(10**11, 10**14, 4))
plt.figure()
for j, B in enumerate(Bs):
    mdotsin = []
    mu = magnetic_moment(B, ns.R_NS)
    for Mdot in Mdots:
        Rsph = spherization_radius_poutanen(Mdot /ns.Medd, ns.Risco, e_wind)
        mass_transfered = mass_transfer_rate_mag_radius(Mdot, mu, ns.Medd, Rsph, ns.M, e_wind=e_wind)
        mdotsin.append(mass_transfered)
    plt.plot(mdots, mdotsin / (Mdots), label="$B$ = 10$^{%d}$" % (np.log10(B)), ls=lss[j])

plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(direction="in", which="both")
plt.ylim(0.4, 1.09)
plt.xlim(1, 10**3.9)
plt.xlabel(r"$\dot{m}_\mathrm{0}$")
plt.ylabel(r"$\dot{M}_\mathrm{R_m}/\dot{M}_\mathrm{0}$")
plt.xscale("log")
plt.legend()
plt.savefig("mushtukov_2019_figure2.png")
