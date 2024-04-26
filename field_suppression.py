from accretion import *
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import time
import argparse

def find_n(Mc=2*10**-4 * u.Msun, Mb=4.6*10**-5 * u.Msun):

        n_a = 1.0001
        n_b = 3
        err_tol = 0.001
        err = (n_b - n_a) / 2
        K = (Mc/Mb)**-2.25

        if K> 0.11:
            raise ValueError("K (%.2f) needs to be below 0.11, Mc needs to be greater than %.1e %s!" % (K, Mc.value, Mc.unit))

        while err > err_tol:
            n = (n_a + n_b) / 2
            f_n = n**-2.25 - n**-3.25 - K
            err = np.abs(n_b - n_a) / 2
            f_na = n_a**-2.25 - n_a**-3.25 - K
            if f_na * f_n <0:
                n_b = n
            else:
                n_a = n
        return n

def magnetospheric_radius_ls(Mdot, B, M_NS=1.4 * M_sun, R_NS=10**6 * u.cm):
    """As given by Middleton+19 accounts for super Edd accretion but no advection"""
    Mdot_17 = Mdot.to(u.g/u.s).value / 10**17
    mu = B.to(u.G) * R_NS.to(u.cm)**3 / (10**30 * u.G * u.cm**3)
    rm = 2.9 * 10**8 * Mdot_17**(-2/7) * (M_NS.to(u.M_sun).value)**(-1/7) * mu**(4/7) * u.cm
    return rm


def mcrit(B):
    """Calculate mdot critical at which the magnetic confiment breaks due to radiation pressure according to Mushtukov
    float: B
        Magnetic field (assumed in G, values are returned in ln(mcrt (g/s)))

    """
    return 6.9233445 + 4.2990807 * np.log(B) - 0.1794699 * np.log(B)**2 + 0.0025782 * np.log(B)**3


def Bfield_decay(B, deltaM, mb=1e-4 * u.M_sun):
    """Returns the new B field"""
    ###deltaM = Mdot.to(u.g/u.s) * deltaT
    return B / (1 + deltaM / mb.to(u.g))


def Bfield_decay_Payne(B, deltaM, mb=4.6 *10**-5 * u.M_sun, mc=2 * 10**-4 * u.M_sun):
    """Returns the new B field after accreting Mdot over delta T.

    Equation 8 in Payne and Melatos 2007
    Equation 35 in Payne 2004

    Parameters
    ----------
    B: astropy.quantity or float
        Initial magnetic field
    """
    #deltaM = Mdot.to(u.Msun/u.s) * deltaT.to(u.s)

    if deltaM < mc / n:
        return B *  (1 - deltaM / mc)
    else:
        return B *  (deltaM / mb) ** (-2.25) # 9/4 = 2.25


def Bfield_decay_Zhang(B, deltaM, Mcrust=0.2 * u.M_sun, xi=1):
    """Returns the new B field

    Parameters
    ----------
    B: astropy.quantity or float
        Initial magnetic field
    """
    x0_2 = (Bf/B)**(4/7)
    C = 1 + np.sqrt(1 - x0_2)
    ##deltaM = Mdot.to(u.g/u.s) * deltaT
    y = 2 * xi * deltaM / (7 * Mcrust.to(u.g))
    return Bf / (1 - (C / np.exp(y) - 1)**2)**(7/4)


def deltaP(torque, deltaT, P, I):

    return (-torque * deltaT * P**2 / (2* np.pi * I)).decompose(bases=[u.s])


def bottom_field(R_NS):
    """Computes the bottom magnetic field according to Zhang & Kojima 2006 Equation 18 but assuming outflows within Rsph and assuming there's no advection
    The magnetospheric radius is taken from Middleton et al. 2023

    Parameters
    ----------
    B: float
        Magnetic field in Gauss
    """
    return (R_NS.to(u.cm).value / (4.2 * 10**7))**(9/4) * 10**12


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Compute decaying field and period due to mass accreton rate in super-critical regime')
    ap.add_argument("-m", "--mdot", nargs='?', help="Mdot in Eddington ratio. Default 10", type=float, default=10)
    ap.add_argument("-t", "--tmax", nargs='?', help="Maximum time in years. Default 1e5 yr", type=float, default=1e5)
    ap.add_argument("-p", "--period", nargs='?', help="Starting period in seconds. Default 10s", type=float, default=10)
    ap.add_argument("-B", "--field", nargs='?', help="Starting magnetic field value in G. Default 10**14 G", type=float, default=10**14)
    ap.add_argument("-d", "--decay", nargs="?", default="Payne", help="Magnetic field decay prescription. Default Payne & Melatos", choices=["Payne", "Zhang", "Shibazaki"])
    args = ap.parse_args()

    home = os.getenv("HOME")

    plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)

    outdir = "field_suppression"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outdir = "%s/mdot_%.1f_P_%.1f_B_%.2E_%s" % (outdir,  args.mdot, args.period, args.field, args.decay)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    decay_keyword = args.decay
    B_init = args.field * u.G
    P_init = args.period * u.s
    M_NS = 1.4 * M_sun
    NS = NS(P_NS = P_init.value, Pso = 0, M=M_NS)
    torque_prescription = torque_wang
    print("Warning: using Wang+95 torque prescription")

    Bf = bottom_field(NS.R_NS)

    print("Bottom magnetic field: %.2E G" % Bf)
    mdot = args.mdot

    print("Running for mdot =  %.1f, P = %.1f s, B = %.2E and Tmax = %.1f yr and %s" % (mdot, args.period, args.field, args.tmax, args.decay))

    #deltaT = 0.5 * u.yr switched to log scale

    #times = np.arange(0, args.tmax, deltaT.to(u.yr).value) * u.yr
    steps = 15000
    print("Number of steps: %d" % steps)

    times = np.geomspace(0.1, args.tmax, steps) # in years
    deltaT = np.diff(times) << u.yr

    B = B_init.value * np.ones(steps) # G
    P_t = P_init * np.ones(steps)
    # variables to be filled
    Mdot_acc = np.zeros(steps) * u.Msun / u.yr
    propeller = np.zeros(steps)
    torques = np.zeros(steps)
    pulsed = np.ones(steps)
    Riscos = np.ones(steps) * NS.Risco.to(u.cm).value
    Macc_accum = 0  * u.Msun

    Mdot = mdot * NS.Medd.to(u.g/u.s) # mdot transferred is constant, even if MdotEdd increases

    if decay_keyword=="Zhang":
        decay_equation = Bfield_decay_Zhang
    elif decay_keyword=="Shibazaki":
        decay_equation = Bfield_decay
    elif decay_keyword == "Payne":
        n = find_n()
        print("Value of n: %.2f" % n)
        decay_equation = Bfield_decay_Payne

    start = time.time()

    for i, t in enumerate(times[1:], 1):
        # update the period so that we get a new Rco and Risco as well as MdotEdd
        NS.P_NS = P_t[i - 1].value # this recalculates Rco internally
        Risco = NS.Risco.value # cm
        Riscos[i] = Risco
        Rsph = 5/3 * mdot * Risco
        Rmag = magnetospheric_radius_ls(Mdot, B[i-1] * u.G, NS.M, NS.R_NS).value # cm

        # SS73 supercritical regime
        if Rmag < Rsph:
            Rmag = 4.2 * 10**7 * (B[i-1] / 10**12) ** (4/9) # cm
            if Rmag < Risco: # non magnetic accretion, no B field decay
                Rmag = Risco
                pulsed[i] = 0

            Mdot_Rm = Mdot * Rmag / Rsph
        # "subcritical" accretion
        else:
            Mdot_Rm = Mdot
        # the angular momentum transfer depends on mdot at Rmag, regardless of the
        # critical value, which is set at the NS
        tau = torque_prescription(Mdot_Rm, Rmag * u.cm, NS.Rco * Risco * u.cm, NS.M)
        # deltaT is not a constant step
        Pincr = deltaP(tau, deltaT[i-1], P_t[i-1], NS.I)
        torques[i] = tau.value
        P_t[i] = P_t[i-1] + Pincr

        if Pincr > 0: # If P increases, we entered propeller, P changes but B does not
            B[i] = B[i-1]
            propeller[i] = True
            Mdot_Rm = 0 * u.g / u.s

        # accretion with magnetosphere (P decreases)
        elif Pincr < 0 and Rmag > Risco:
            # if we have exceeded the critical value, readjust for magnetic field suppression
            critical_mdot = np.exp(mcrit(B[i-1])) * (u.g/u.s)
            Mdot_Rm = critical_mdot if Mdot_Rm.to(u.g/u.s) > critical_mdot else Mdot_Rm
            # add the new matter
            Maccumulated =  Mdot_Rm.to(u.Msun / u.s) * deltaT[i-1].to(u.s) + Macc_accum
            B[i] = decay_equation(B_init.value, Maccumulated) # B[i-1]

        # Rmag at Isco already, there's no B decay
        else:
            B[i] = B[i-1]

        Mdot_acc[i] = Mdot_Rm
        Macc_accum += Mdot_Rm.to(u.Msun / u.s) * deltaT[i-1].to(u.s)

        print("Progress: %d/%d" % (i, steps), end="\r")

    outputs = np.array([B[:-1], P_t[:-1], Mdot_acc[:-1].to(u.Msun / u.yr), torques[1:], propeller[1:], pulsed[1:], times[:-1]])
    np.savetxt("%s/mdot_%.1f_P_%.1f_B_%.2E.dat" %(outdir, mdot, P_init.value, args.field), outputs.T, delimiter="\t",
               fmt="%.5E", header="B\tP\tMacc\ttorque_cm2_g_s2\tpropeller\tpulsed\tt")

    end = time.time()
    time_taken = end - start

    print("Done", "\n")

    print("Time taken: %.2f s" % time_taken)

    print("Adding up mdot")
    M_acc = np.cumsum(Mdot_acc[:-1] *  deltaT)
    fig, axes = plt.subplots(4, 1, sharex=True, gridspec_kw={"hspace":0.2})
    ax = axes[0]
    scaling = args.tmax
    ax.plot(times[1:-1] / scaling, B[1:-1])
    #ax.fill_between(times[:-1] / scaling, 0, 1, where=propeller[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
    #ax.fill_between(times[:-1] / scaling, 0, 1, where=~pulsed[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
    ax.set_yscale("log")
    ax.margins(y=0.025)
    ax.set_xscale("log")
    ax.set_ylabel("B (G)")

    ax = axes[1]
    ax.plot(times[1:-1] / scaling, P_t[1:-1])
    ax.set_yscale("log")
    ax.set_ylabel("P(s)")
    ax.set_xscale("log")

    ax = axes[2]
    ax.plot(times[1:-1] / scaling, Mdot_acc[1:-1].to(u.M_sun / u.yr))
    ax.set_yscale("log")
    ax.set_ylabel('$\dot{M}$ ($M_\odot$ / yr)')
    ax.set_xscale("log")

    ax = axes[3]
    ax.plot(times[1:-1] / scaling, M_acc[1:])
    ax.set_yscale("log")
    ax.set_ylabel('Mass Acc ($M_\odot$)')
    ax.set_xscale("log")
    ax.set_xlabel("Time (%.0e yr)" % scaling)
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.savefig("%s/multiplot.png" % (outdir))

    plt.figure()
    plt.scatter(times / scaling, Riscos / 1000) # convert to km
    plt.ylabel("$R_{isco}$ (km)")
    plt.xlabel("Time (%.0e yr)" % (scaling))
    plt.xscale("log")
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.savefig("%s/isco.png" % (outdir))

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
    print("Save to %s" % outdir)
