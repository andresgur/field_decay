from compact_object import NS, magnetospheric_radius, torque_wang, Gcgs, ccgs, M_sun, magnetospheric_radius_superEdd
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import time
import argparse
from math import cos, sin
import numexpr as ne
from celerite.terms import RealTerm
from celerite import GP

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


def mcrit(B):
    """Calculate mdot critical at which the magnetic confiment breaks due to radiation pressure according to Mushtukov
    float: B
        Magnetic field (assumed in G, values are returned in ln(mcrt (g/s)))

    """
    return ne.evaluate("6.9233445 + 4.2990807 * log(B) - 0.1794699 * log(B)**2 + 0.0025782 * log(B)**3")


def Bfield_decay(B, deltaM, mb=1e-4 * u.M_sun):
    """Returns the new B field
    
    deltaM: float,
        Mass accreted in g
    """
    return B / (1 + deltaM / mb.to(u.g).value)


def Bfield_decay_Payne(B, deltaM, mb=4.6 *10**-5 * u.M_sun, mc=2 * 10**-4 * u.M_sun):
    """Returns the new B field after accreting Mdot over delta T.

    Equation 8 in Payne and Melatos 2007
    Equation 35 in Payne 2004

    Parameters
    ----------
    B: float
        Initial magnetic field
    deltaM: float,
        Mass accreted in g
    """

    if deltaM < mc.to(u.g).value / n:
        return B *  (1 - deltaM / mc.to(u.g).value)
    else:
        return B *  (deltaM / mb.to(u.g).value) ** (-2.25) # 9/4 = 2.25


def Bfield_decay_Zhang(B, deltaM, Mcrust=0.2 * u.M_sun, xi=1):
    """Returns the new B field

    Parameters
    ----------
    B: float
        Initial magnetic field
    deltaM: float,
     Mass accreted in g
    """
    x0_2 = (Bf/B)**(4/7)
    C = 1 + np.sqrt(1 - x0_2)
    ##deltaM = Mdot.to(u.g/u.s) * deltaT
    y = 2 * xi * deltaM / (7 * Mcrust.to(u.g).value)
    return Bf / (1 - (C / np.exp(y) - 1)**2)**(7/4)


def propeller_torque(Mdot, M_NS, omega, Rm):
    """Equation 12 from Illarionov & Sunyaev 1975 or Eq 42 from Abolmasov 2024 review
    
    omega: float,
        Angular velocity of the NS (2 pi / P)
    Rm :float,
        Magnetospheric radius
    """
    return ne.evaluate("-Mdot * Gcgs * M_NS / Rm / omega")


def deltaP(torque, deltaT, P, I):
    """ Change in the spin period
    deltaT: float,
        Time step (in seconds)

    """
    two_pi = 2*np.pi
    return ne.evaluate("-torque * deltaT * P**2 / (two_pi * I)")


def deltaangle(torque, deltaT, omega, I):
    """Change in the axis angle. Works for both alpha or chi
    omega: float,
    Angular velocity of the NS (2 pi / P)
    """
    return ne.evaluate("torque * deltaT / (omega * I)")


def braking_torque(mu, Rlc, chi=np.pi / 4):
    """Equation from Biryukov and Abolmasov 2021
    
    mu: float,
        Magnetic moment
    Rlc: float,
        Light cylinder
    chi: float,
        Magnetic angle (radiants)
    """
    return ne.evaluate("-mu**2/ Rlc**3")

def cos_function(eta, alpha, chi):
    """Equation 20 combined with eta (see Eq 26) from Byryukov and Abolmasov 2021"""
    A = (1 - eta/2 * (sin(chi)**2 * sin(alpha)**2 + 2 *cos(chi)**2 * cos(alpha)**2))**-1
    return eta* A



def bottom_field(R_NS):
    """Computes the bottom magnetic field according to Zhang & Kojima 2006 Equation 18 but assuming outflows within Rsph and assuming there's no advection
    The magnetospheric radius is taken from Middleton et al. 2023

    Parameters
    ----------
    R_NS:float,
        Radius of the NS in cm
    B: float
        Magnetic field in Gauss
    """
    return (R_NS / (4.2 * 10**7))**(9/4) * 10**12


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Compute decaying field and period due to mass accreton rate in super-critical regime')
    ap.add_argument("-m", "--mdot", nargs='?', help="Mdot in Eddington ratio. Default 10", type=float, default=10)
    ap.add_argument("-t", "--tmax", nargs='?', help="Maximum time in years. Default 1e5 yr", type=float, default=1e5)
    ap.add_argument("-p", "--period", nargs='?', help="Starting period in seconds. Default 10s", type=float, default=10)
    ap.add_argument("-B", "--field", nargs='?', help="Starting magnetic field value in G. Default 10**14 G", type=float, default=10**14)
    ap.add_argument("-c", "--chi", nargs='?', help="Magnetic angle (degrees)", type=float, default=55)
    ap.add_argument("-a", "--alpha", nargs='?', help="Spin axis angle (degrees)", type=float, default=45)
    ap.add_argument("-d", "--decay", nargs="?", default="Payne", help="Magnetic field decay prescription. Default Payne & Melatos", choices=["Payne", "Zhang", "Shibazaki"])
    args = ap.parse_args()

    home = os.getenv("HOME")
    stylefile = '%s/.config/matplotlib/stylelib/paper.mplstyle' % home
    if os.path.isfile(stylefile):
        plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)

    decay_keyword = args.decay
    B_init = args.field 
    P_init = args.period
    M_NS = 1.4 * M_sun
    NS = NS(P_NS = P_init, Pso = 0, M_NS=M_NS, chi=args.chi / 180 * np.pi, alpha=args.alpha / 180 * np.pi)
    accretion_torque = torque_wang # use Wang's prescription for the torque

    Bf = bottom_field(NS.R_NS)

    print("Bottom magnetic field: %.2E G" % Bf)
    mdot = args.mdot

    print("Running for mdot =  %.1f, P = %.1f s, B = %.2E, spin-axis angle: %.1f, magnetic angle: %.1f, Tmax = %.1f yr decay law: %s" % (mdot, args.period, args.field, args.alpha, args.chi, args.tmax, args.decay))

    #deltaT = 0.5 * u.yr switched to log scale

    #times = np.arange(0, args.tmax, deltaT.to(u.yr).value) * u.yr
    nsteps = 50000
    print("Number of steps: %d" % nsteps)
    tswitch = 2.5e4
    #times = np.concatenate((np.geomspace(0.1, tswitch, nsteps), np.linspace(tswitch, args.tmax, 2 * nsteps))) # in years
    times = np.geomspace(0.1, args.tmax, nsteps) # # in years
    #times = np.arange(0.1, args.tmax, )
    nsteps = len(times)
    steps = np.arange(1, nsteps + 1)
    
    deltaT = (np.diff(times) << u.yr).to(u.s).value
    print(f"Delta T early: {deltaT[0]:.1f}")
    print(f"Delta T late: {deltaT[-1]:.1f}")
    #print(f"Delta T (s):{deltaT:.2f}")

    # variables to be filled
    B = B_init * np.ones(nsteps) # G
    P_t = P_init * np.ones(nsteps)
    Mdot_t = np.zeros(nsteps) # cgs
    Mcrits = np.zeros(nsteps)
    propeller = np.zeros(nsteps)
    T_spin_t = np.zeros(nsteps)
    T_alpha_t = np.zeros(nsteps)
    T_chi_t = np.zeros(nsteps)
    pulsed = np.ones(nsteps)
    Riscos = np.ones(nsteps) * NS.Risco
    Rmags = np.ones(nsteps)
    Rsphs = np.ones(nsteps)
    Rcor = np.ones(nsteps)
    alpha_t = np.ones(nsteps) * NS.alpha
    chi_t = np.ones(nsteps) * NS.chi
    T_propeller_t = np.zeros(nsteps)
    mdots = mdot  * np.ones(nsteps)
    # mean of mdot, 1 day bendtimescale, variance = 1 mdot
    timescale = 10 / 365

    np.random.seed(15)
    kernel = RealTerm(log_a=np.log(36), log_c=np.log(2 * np.pi / timescale)) 
    gp = GP(kernel, mean=mdot)
    gp.compute(times)
    mdots = gp.sample()
    #print(mdots[np.isnan(mdots)])
    #mdots[np.isnan(mdots)] = mdot
    eta = 0.99

    Macc_accum = 0 # u.g/u.s

    Mdot = mdots * NS.Medd # mdot transferred is constant, even if MdotEdd increases, g/s

    if decay_keyword=="Zhang":
        decay_equation = Bfield_decay_Zhang
    elif decay_keyword=="Shibazaki":
        decay_equation = Bfield_decay
    elif decay_keyword == "Payne":
        n = find_n()
        print("Value of n: %.2f" % n)
        decay_equation = Bfield_decay_Payne

    start = time.time()
    try:
        #for i, t in enumerate(times[1:], 1):
        for i in steps[:-1]:
            # update the period so that we get a new Rco and Risco as well as MdotEdd
            NS.P_NS = P_t[i - 1] # this recalculates Rco internally
            critical_mdot = np.exp(mcrit(B[i-1])) # cgs
            Mcrits[i - 1] = critical_mdot
            Riscos[i] = NS.Risco if NS.Risco > NS.R_NS else NS.R_NS # cm
            Rsph = 5/3 * mdots[i -1] * NS.Risco # cm
            Rsphs[i] = Rsph
            Rmag = magnetospheric_radius(Mdot[i-1], B[i-1], NS.M, NS.R_NS) # cm

            # SS73 supercritical regime
            if Rmag < Rsph:
                Rmag = magnetospheric_radius_superEdd(NS, B[i-1])#4.2 * 10**7 * (B[i-1] / 10**12) ** (4/9) # cm
                if Rmag < NS.Risco: # non magnetic accretion, no B field decay
                    Rmag = NS.Risco
                    pulsed[i] = 0

                Mdot_Rm = Mdot[i-1] * Rmag / Rsph
            # "subcritical" accretion
            else:
                Mdot_Rm = Mdot[i-1]

            Rmags[i] = Rmag
            Rcor[i] = NS.rco * NS.Risco
            # propeller
            if Rcor[i] < Rmags[i]:
                propeller[i] = 1
                T_propeller_t[i - 1] = propeller_torque(Mdot_Rm, NS.M, NS.omega, Rmag)
                T_spin_t[i-1] += T_propeller_t[i-1] * cos(alpha_t[i - 1])
                T_chi_t[i -1] += cos_function(eta, alpha_t[i-1], chi_t[i-1]) * T_propeller_t[i-1] * sin(alpha_t[i -1 ])**2 * cos(alpha_t[i - 1]) * sin(chi_t[i - 1]) * cos(chi_t[i-1])
                B[i] = B[i-1]
            # accretion
            else:
                T_accretion = accretion_torque(Mdot_Rm, Rmag, Rcor[i], NS.M)
                T_spin_t[i - 1] += T_accretion * cos(alpha_t[i - 1])
                T_alpha_t[i -1] += -T_accretion * sin(alpha_t[i - 1])
                T_chi_t[i -1] += cos_function(eta, alpha_t[i-1], chi_t[i-1]) * T_accretion * sin(alpha_t[i -1 ])**2 * cos(alpha_t[i - 1]) * sin(chi_t[i - 1]) * cos(chi_t[i-1])

                # accretion with magnetosphere --> B decays
                if Rmag > NS.Risco:
                    # if we have exceeded the critical value, readjust for magnetic field suppression
                    Mdot_t[i - 1] = critical_mdot if Mdot_Rm > critical_mdot else Mdot_Rm
                    # add the new matter to decay the B field
                    Maccumulated = Mdot_t[i - 1] * deltaT[i-1] + Macc_accum # cgs
                    B[i] = decay_equation(B_init, Maccumulated) # B[i-1]
                # Rmag at Isco already, there's no B decay
                else:
                    B[i] = B[i-1]
                    Mdot_t[i -1] = Mdot_Rm
                
                Macc_accum += Mdot_t[i -1] * deltaT[i-1] # cgs
            
            mu = B[i-1] * NS.R_NS**3 / 2
            T_spin_t[i - 1] += braking_torque(mu, NS.rlc * NS.Risco) * (1 + sin(chi_t[i - 1])**2)
            T_chi_t[i -1] += braking_torque(mu, NS.rlc * NS.Risco) * sin(chi_t[i- 1 ]) * cos(chi_t[i - 1])
            # update spin period
            Pincr = deltaP(T_spin_t[i - 1], deltaT[i-1], P_t[i-1], NS.I)
            P_t[i] = P_t[i-1] + Pincr
            # update alpha
            alpha_incr = deltaangle(T_alpha_t[i -1], deltaT[i-1], NS.omega, NS.I)
            alpha_t[i] = alpha_t[i-1] + alpha_incr
            # update chi
            chi_incr = deltaangle(T_chi_t[i -1], deltaT[i-1], NS.omega, NS.I)
            chi_t[i] =  chi_t[i-1] + chi_incr

            print("Progress: %d/%d" % (i, nsteps), end="\r")
    except ValueError as e:
        print("Math overflow. Aborting program", e)
        B[i:] = B[i - 1]
        P_t[i:] = P_t[i - 1]
        alpha_t[i:] = alpha_t[i - 1]
        chi_t[i:] = chi_t[i - 1]

    outdir = "field_suppression"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outdir = "%s/mdot_%.1f_P_%.1f_B_%.2E_%s_Chi_%.1f_a_%.1f" % (outdir,  args.mdot, args.period, args.field, args.decay, args.chi, args.alpha)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    conversion = ((1 * (u.g/u.s)).to(u.Msun / u.yr)).value
    
    outputs = np.array([B[:-1], P_t[:-1], Mdot_t[:-1] * conversion, Mcrits[1:] * conversion, Rmags[1:] / 1000, Rsphs[1:] / 1000, Rcor[1:] / 1000, T_spin_t[1:], alpha_t[1:] / np.pi * 180, chi_t[1:] / np.pi * 180, propeller[1:], pulsed[1:], times[:-1]])
    np.savetxt("%s/results.dat" %(outdir), outputs.T, delimiter="\t",
               fmt="%.5E\t%.5f\t%.5E\t%.5E\t%.1f\t%.1f\t%.1f\t%.2E\t%.2f\t%.2f\t%d\t%d\t%.5E", header="B\tP\tMacc\tMcrit\tRmag_km\tRsph_km\tRcor_km\ttorque_cm2_g_s2\talpha\tchi\tpropeller\tpulsed\tt")

    end = time.time()
    time_taken = end - start

    print("Done", "\n")

    print("Time taken: %.2f s" % time_taken)

    print("Adding up mdot")
    M_acc = (np.cumsum(Mdot_t[:-1] *  deltaT) * u.g).to(u.M_sun).value # convert to solar masses

    fig, axes = plt.subplots(5, 1, sharex=True, gridspec_kw={"hspace":0.15}, figsize=(18, 14))
    i = 0
    ax = axes[i]
    scaling = 1
    plot_times = times[1:-1] / scaling
    ax.plot(plot_times, B[1:-1])
    #ax.fill_between(times[:-1] / scaling, 0, 1, where=propeller[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
    #ax.fill_between(times[:-1] / scaling, 0, 1, where=~pulsed[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
    ax.set_yscale("log")
    ax.margins(y=0.025)
    ax.set_xscale("log")
    ax.set_ylabel("B (G)")

    ax2 = ax.twinx()
    ax = ax2#axes[1]
    color = "C1"
    ax.plot(plot_times, P_t[1:-1], color=color, ls="--", lw=1.5)
    ax.tick_params("y", colors=color, which="both")
    ax.spines["right"].set_color(color)
    ax.yaxis.label.set_color(color)
    ax.set_ylabel("P(s)")
    ax.set_yscale("log")
    ax.set_xscale("log")

    i+=1
    ax = axes[i]
    ax.plot(plot_times, Mdot[1:-1] * conversion * 10**6, label=r"$\dot{M}_0$", ls="--", color="black", alpha=0.9, lw=1)
    ax.plot(plot_times, Mdot_t[1:-1] * conversion * 10**6, label=r"$\dot{M}_\mathrm{NS}$")
    ax.plot(plot_times, Mcrits[1:-1] * conversion * 10**6, label=r"$\dot{M}_\mathrm{crit}$", ls="--")
    ax.legend()
    ax.set_ylabel('$\dot{M}$ (10$^{-6}$ $M_\odot$/yr)')
    ax.set_xscale("log")
    i+=1
    ax = axes[i]
    ax.plot(plot_times, M_acc[1:])
    ax.set_ylabel(r'$M_\mathrm{a}$ ($M_\odot$)')
    ax.set_xscale("log")
    ax.set_yscale("log")
    i+=1
    ax = axes[i]
    ax.plot(plot_times, Rmags[1:-1] / 1000, label=r"$R_\mathrm{mag}$")
    ax.plot(plot_times, Rsphs[1:-1]  / 1000, label=r"$R_\mathrm{sph}$", ls="--", lw=1)
    ax.plot(plot_times, Rcor[1:-1]  / 1000, label=r"$R_\mathrm{cor}$", ls=":")
    #ax.axhline(NS.R_NS / 1000, label=r"$R_\mathrm{NS}$", color="black", ls="--", alpha=0.9)
    ax.legend()
    ax.set_ylabel('R (km)')
    ax.set_xscale("log")
    ax.set_yscale("log")

    i+=1
    ax = axes[i]
    ax.plot(plot_times, alpha_t[1:-1] / np.pi * 180, label=r"$\alpha$")
    ax.plot(plot_times, chi_t[1:-1] / np.pi * 180, label=r"$\chi$", ls="--")
    ax.legend()
    ax.set_ylabel(r'Angle ($^\circ$)')
    ax.set_xscale("log")

    axes[-1].set_xlabel("Time (yr)" )
    plt.savefig("%s/multiplot.png" % (outdir))

    plt.figure()
    plt.scatter(times / scaling, Riscos / 1000) # convert to km
    plt.axhline(NS.R_NS / 1000, label=r"$R_\mathrm{NS}$", color="black", ls="--", alpha=0.9)
    plt.legend()
    plt.ylabel("$R_{isco}$ (km)")
    plt.xlabel("Time (%.0e yr)" % (scaling))
    plt.xscale("log")
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.savefig("%s/isco.png" % (outdir), dpi=100)


    plt.figure()
    nudot = np.diff(1 / P_t) / deltaT
    plt.scatter(times[1:-1] / scaling, nudot[1:]) # convert to km
    plt.ylabel(r"$\dot{\nu}$ (Hz/s)")
    plt.xlabel("Time (yr)")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    #ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.savefig("%s/dpdt.png" % (outdir), dpi=100)

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
