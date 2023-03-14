from formulae.accretion import *
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import time
import argparse

def mcrit(B):
    """Calculate mdot critical at which the magnetic confiment breaks due to radiation pressure according to Mushtukov
    float: B
        Magnetic field (assumed in G, values are returned in ln(mcrt (g/s)))

    """
    return 6.9233445 + 4.2990807 * np.log(B) - 0.1794699 * np.log(B)**2 + 0.0025782 * np.log(B)**3


ap = argparse.ArgumentParser(description='Compute decaying field and period due to mass accreton rate in super-critical regime')

ap.add_argument("-m", "--mdot", nargs='?', help="Mdot in Eddington ratio", type=float, default=10)
ap.add_argument("-t", "--tmax", nargs='?', help="Maximum time in years", type=float, default=1e5)
ap.add_argument("-p", "--period", nargs='?', help="Starting period in seconds", type=float, default=10)
ap.add_argument("-B", "--field", nargs='?', help="Starting magnetic field value", type=float, default=10**14)
args = ap.parse_args()


home = os.getenv("HOME")

plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)

outdir = "field_suppression"
if not os.path.isdir(outdir):
    os.mkdir(outdir)


def Bfield(B, Mdot, deltaT, mb=1e-4 * u.M_sun):

    return B / (1 + Mdot.to(u.g/u.s) * deltaT.to(u.s) / mb.to(u.g))


def deltaP(torque, deltaT, P, I):

    return (-torque * deltaT * P**2 / (2* np.pi * I)).decompose(bases=[u.s])


B_init = args.field * u.G
P_init = args.period * u.s
M_NS = 1.4 * M_sun
NS = NS(P_NS = P_init.value, Pso = 0, M=M_NS)

mdot = args.mdot

print("Running for mdot =  %.1f, P = %.1f s, B = %.2E and Tmax = %.1f yr" % (mdot, args.period, args.field, args.tmax))
deltaT = 1 * u.yr

times = np.arange(0, args.tmax, deltaT.to(u.yr).value) * u.yr

steps = len(times)
print("Number of steps: %d" % steps)
B = B_init * np.ones(steps)
P_t = P_init * np.ones(steps)
# variables to be filled
Mdot_acc = np.zeros(steps) * u.Msun / u.yr
propeller = np.ones(steps)
start = time.time()
pulsed = np.ones(steps)
Riscos = np.ones(steps) * NS.Risco.unit
Riscos[0] = NS.Risco


for i, t in enumerate(times[1:], 1):
    # update the period so that we get a need Rco and Risco as well as Mdot
    NS.P_NS(P_t[i - 1].value) # this recalculates Rco
    Risco = NS.Risco
    Riscos[i] = Risco
    Mdot = mdot * NS.Medd.to(u.g/u.s)
    Rsph = 5/3 * mdot * NS.Risco
    Rmag = magnetospheric_radius_ls(Mdot, B[i-1], NS.M, NS.R_NS)

    # SS73 regime
    if Rmag < Rsph:
        Rmag = 4.2 * 10**7 * (B[i-1].to(u.G).value / 10**12) ** (4/9) * u.cm
        if Rmag < Risco: # non magnetic accretion, B field decay
            Rmag = Risco
            pulsed[i] = 0

        Mdot_Rm = Mdot * Rmag / Rsph
    # "subcritical" accretion
    else:
        Mdot_Rm = Mdot
    # the angular momentum transfer depends on mdot at Rmag, regardless of the critical value, which is set at the NS
    tau = torque(Mdot_Rm, Rmag, NS.Rco * Risco, NS.M)
    Pincr = deltaP(tau, deltaT, P_t[i-1], NS.I)
    P_t[i] = Pincr + P_t[i-1]

    if tau < 0: # If tau is negative, we entered propeller, P changes but B does not
        B[i] = B[i-1]
        propeller[i] = True
        Mdot_acc[i] = 0
    # accretion with magnetosphere
    elif tau > 0 and Rmag > Risco:
        propeller[i]= False
        # if we have exceeded the critical value, readjust for magnetic field suppression
        critical_mdot = np.exp(mcrit(B[i-1].value)) * (u.g/u.s)
        Mdot_Rm = critical_mdot if Mdot_Rm.to(u.g/u.s) > critical_mdot else Mdot_Rm
        B[i] = Bfield(B[i-1], Mdot_Rm, deltaT)
        Mdot_acc[i] = Mdot_Rm
    # Rmag at Isco already
    else:
        B[i] = B[i-1]
        Mdot_acc[i] = Mdot_Rm

    print("Progress: %d/%d" % (i, steps), end="\r")

outputs = np.array([B[:-1], P_t[:-1], Mdot_acc[:-1].to(u.Msun / u.yr), times[:-1]])
np.savetxt("%s/mdot_%.1f_P_%.1f_B_%.2E.dat" %(outdir, mdot, P_init.value, args.field), outputs.T, delimiter="\t",
           fmt="%.5E", header="B\tP\tMacc\tt")

end = time.time()
time_taken = end - start

print("Time taken: %.2f s" % time_taken)
print("Adding up mdot")
M_acc = np.cumsum(Mdot_acc *  deltaT)
print("Done")
fig, axes = plt.subplots(4, 1, sharex=True, gridspec_kw={"hspace":0.2})
ax = axes[0]
scaling = args.tmax
ax.plot(times[:-1].value / scaling, B[:-1])
#ax.fill_between(times[:-1] / scaling, 0, 1, where=propeller[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
#ax.fill_between(times[:-1] / scaling, 0, 1, where=~pulsed[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
ax.set_yscale("log")
ax.margins(y=0.025)
ax.set_xscale("log")
ax.set_ylabel("B (G)")

ax = axes[1]
ax.plot(times[:-1].value / scaling, P_t[:-1])
ax.set_yscale("log")
ax.set_ylabel("P(s)")
ax.set_xscale("log")

ax = axes[2]
ax.plot(times[:-1].value / scaling, Mdot_acc[:-1].to(u.M_sun / u.yr))
ax.set_yscale("log")
ax.set_ylabel('$\dot{M}$ ($M_\odot$ / yr)')
ax.set_xscale("log")

ax = axes[3]
ax.plot(times[:-1].value / scaling, M_acc[:-1])
ax.set_yscale("log")
ax.set_ylabel('Mass Acc ($M_\odot$)')
ax.set_xscale("log")
ax.set_xlabel("Time (10$^5$ yr)")
ax.set_yscale("log")
ax.set_xscale("log")
plt.savefig("%s/mdot_%.1f_P_%.1f_B_%.2E.png" % (outdir, mdot, P_init.value, args.field))

plt.figure()
plt.scatter(times.value/ scaling, Riscos.to(u.km).value)
plt.ylabel("$R_{isco}$ (km)")
plt.xlabel("Time (10$^5$ yr)")
plt.xscale("log")
plt.savefig("%s/isco_%.1f_P_%.1f_B_%.2E.png" % (outdir, mdot, P_init.value, args.field))

if False:
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlabel('Mass Accreted ($M_\odot$)')
    xticks2 = ax.get_xticks()
    ax2.set_xticks(xticks2)

    indexes = [np.argmin(np.abs(times.value / scaling - t)) for t in xticks2]
    maccs = (Mdot_acc[indexes].to(u.Msun)).value
    xticks2labels = ["%.1E" % (m) for m in maccs]
    ax2.set_xticklabels(xticks2labels)
    ax2.set_xlim(ax.get_xlim())
