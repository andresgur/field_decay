from compact_object import NS, Gcgs, ccgs, M_sun
from accretion import accretion_torque_dai, magnetic_torque_dai, magnetic_torque_dai_propeller, magnetospheric_radius_superEdd, magnetospheric_radius, mcrit
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import time
import argparse
from math import cos, sin, log, exp, pi
import numexpr as ne
from celerite.terms import RealTerm
from celerite import GP
from tqdm import tqdm
from field_decay_law import ShibazakiFieldDecay, PayneFieldDecay, ZhangFieldDecay
import shutil

def read_config(file):
    config = {}
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Split the line into key-value pairs
            linesplit = line.split()
            key = linesplit[0]
            values = linesplit[1:]
            try:
                config[key] = [float(value) for value in values]
            except (ValueError, SyntaxError):
                config[key] = values[0] # String for the decay laws

    return config


def deltaP(torque, deltaT, P, I):
    """ Change in the spin period
    deltaT: float,
        Time step (in seconds)

    """
    return -torque * deltaT * P**2 / (2 * pi * I)


def deltaangle(torque, deltaT, omega, I):
    """Change in the axis angle. Works for both alpha or chi
    omega: float,
    Angular velocity of the NS (2 pi / P)
    """
    return torque * deltaT / (omega * I)


def braking_torque(mu, Rlc):
    """e.g. Equation 14 from Biryukov and Abolmasov 2021
    
    mu: float,
        Magnetic moment
    Rlc: float,
        Light cylinder
    chi: float,
        Magnetic angle (radiants)
    """
    return -mu**2/ Rlc**3

def cos_function(eta, alpha, chi):
    """Equation 20 combined with eta (see Eq 26) from Byryukov and Abolmasov 2021"""
    A = (1 - eta/2 * (sin(chi)**2 * sin(alpha)**2 + 2 *cos(chi)**2 * cos(alpha)**2))**-1
    return eta * A

conversion = ((1 * (u.g/u.s)).to(u.Msun / u.yr)).value

DECAY_LAWS = ["Payne", "Shibazaki", "Zhang"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Compute decaying field and period due to mass accreton rate in super-critical regime')
    ap.add_argument("--config", nargs="?", help="Config file")
    args = ap.parse_args()

    home = os.getenv("HOME")
    stylefile = '%s/.config/matplotlib/stylelib/paper.mplstyle' % home
    if os.path.isfile(stylefile):
        plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)
    
    parameters = read_config(args.config)
    if parameters["law"] in DECAY_LAWS:
        decay_keyword =  parameters["law"]
    else:
        raise ValueError("Decay keyword %s not valid! Available options are: %s" % (parameters["law"], " ".join(DECAY_LAWS)))

    B_inits = parameters["B"]
    P_inits = parameters["P"]
    mdots = parameters["mdot"]
    M_NS = parameters["M"][0] * u.M_sun
    R_NS = parameters["R"][0] * u.cm
    chi = parameters["chi"][0]
    alpha = parameters["alpha"][0]
    eta = parameters["eta"][0]
    # steps 
    nsteps = int(parameters["steps"][0])
    tmax = parameters["tmax"][0]

    #deltaT = 0.5 * u.yr switched to log scale

    #times = np.arange(0, tmax, deltaT.to(u.yr).value) * u.yr
    #nsteps = 100000 #// 4# 50000
    print("Number of steps: %d" % nsteps)
    tswitch = 2.5e4
    #times = np.concatenate((np.geomspace(0.1, tswitch, nsteps), np.linspace(tswitch, tmax, 2 * nsteps))) # in years
    times = np.geomspace(0.1, tmax, nsteps) # # in years
    #times = np.arange(0.1, tmax, )
    nsteps = len(times)
    steps = range(nsteps)
    
    deltaT = (np.diff(times) << u.yr).to(u.s).value
    print(f"Delta T early: {deltaT[0]:.1f}")
    print(f"Delta T late: {deltaT[-1]:.1f}")
    for mdot in mdots:
        for P_init in P_inits:
            for B_init in B_inits:

                print("Running for mdot =  %.1f, P = %.3f s, B = %.2E, spin-axis angle: %.1f" \
                    ", magnetic angle: %.1f, Tmax = %.1f yr decay law: %s" % (mdot, P_init, B_init, alpha, chi, tmax, decay_keyword))

                neutron_star = NS(P_NS = P_init, Pso = 0, M_NS=M_NS, R_NS=R_NS, chi=chi / 180 * np.pi, alpha=alpha / 180 * np.pi)
                #print(f"Delta T (s):{deltaT:.2f}")

                # variables to be filled
                B = B_init * np.ones(nsteps) # G
                P_t = P_init * np.ones(nsteps)
                Mdot_NS = np.zeros(nsteps)
                Mdot_Rmag = np.zeros(nsteps)
                Mcrits = np.zeros(nsteps)
                propeller = np.zeros(nsteps)
                T_spin_t = np.zeros(nsteps)
                T_alpha_t = np.zeros(nsteps)
                T_chi_t = np.zeros(nsteps)
                pulsed = np.ones(nsteps)
                Rins = np.ones(nsteps) * neutron_star.Risco
                Rmags = np.ones(nsteps)
                Rsphs = np.ones(nsteps)
                Rcor = np.ones(nsteps) * neutron_star.Rco
                alpha_t = np.ones(nsteps) * neutron_star.alpha
                chi_t = np.ones(nsteps) * neutron_star.chi
                T_disc = np.zeros(nsteps)
                T_mag = np.zeros(nsteps)
                T_brake = np.zeros(nsteps)
                mdots = mdot  * np.ones(nsteps)
                # mean of mdot, 1 day bendtimescale, variance = 1 mdot
                timescale = 10 / 365

                np.random.seed(15)
                kernel = RealTerm(log_a=log(10), log_c=log(2 * pi / timescale)) 
                gp = GP(kernel, mean=mdot)
                gp.compute(times)
                #mdots = gp.sample()
                #print(mdots[np.isnan(mdots)])
                #mdots[np.isnan(mdots)] = mdot

                Macc_accum = 0 # u.g/u.s

                Mdot = mdots * neutron_star.Medd # mdot transferred is constant, even if MdotEdd increases, g/s
                print("Mdot: %.1e g/s (%.1e M_sun/yr)" % (Mdot[0], Mdot[0] * conversion))

                if decay_keyword=="Zhang":
                    decay_law = ZhangFieldDecay(B_init)
                    print("Bottom magnetic field: %.2E G" % decay_law.Bf)
                elif decay_keyword=="Shibazaki":
                    decay_law = ShibazakiFieldDecay(B_init)
                elif decay_keyword == "Payne":
                    decay_law = PayneFieldDecay(B_init)
                    print("Value of n: %.2f" % decay_law.n)
                print("Using decay law from: %s" % decay_law.name)
                start = time.time()
                try:
                    #for i, t in enumerate(times[1:], 1):
                    for i in tqdm(steps[:-1]):
                        # update the period so that we get a new Rco and Risco as well as MdotEdd
                        neutron_star.P_NS = P_t[i] # this recalculates Rco internally
                        critical_mdot = exp(mcrit(B[i])) # cgs
                        Mcrits[i] = critical_mdot
                        Rins[i] = neutron_star.Risco if neutron_star.Risco > neutron_star.R_NS else neutron_star.R_NS # cm
                        Rcor[i] = neutron_star.Rco
                        Rsph = 5/3 * mdots[i] * Rins[i] # cm this is the NS radius or Rin (see Lipunova 1999 how the inner torque is defined)
                        Rsphs[i] = Rsph
                        #Rmag, err = magnetospheric_radius_wang(Mdot[i-1], B[i-1], NS) # cm
                        Rmag = magnetospheric_radius(Mdot[i], B[i], neutron_star, psi=0.5)
                        # SS73 supercritical regime
                        if Rmag < Rsph:
                            #Rmag = magnetospheric_radius_wang_superEdd(Mdot[i-1], B[i-1], NS)#4.2 * 10**7 * (B[i-1] / 10**12) ** (4/9) # cm
                            Rmag = magnetospheric_radius_superEdd(B[i], neutron_star, psi=0.5)
                            # for numerical stability (mainly with Payne's mangetic field decay) let's allow some tolerance
                            #if Rins[i] - TOLERANCE < Rmag < Rins[i] + TOLERANCE: # non magnetic accretion, no B field decay
               

                            if (Rmag <= Rins[i]):
                                Rmag = Rins[i]
                                pulsed[i] = 0

                            Mdot_Rmag[i] = Mdot[i] * Rmag / Rsph
                        # "subcritical" accretion
                        else:
                            Mdot_Rmag[i] = Mdot[i]

                        Rmags[i] = Rmag
                        T_disc[i] = accretion_torque_dai(Mdot_Rmag[i], Rmag, neutron_star) # this torque works for both accretion and propeller
                        # propeller (remember torque = 0 is spin eq, not propeller)
                        if Rcor[i] < Rmags[i]:
                            propeller[i] = 1
                            B[i + 1] = B[i]
                            magnetic_torque = magnetic_torque_dai_propeller
                        # accretion
                        else:
                            magnetic_torque = magnetic_torque_dai
                            # accretion with magnetosphere --> B decays
                            if pulsed[i]:
                                # if we have exceeded the critical value, readjust for magnetic field suppression
                                Mdot_NS[i] = critical_mdot if Mdot_Rmag[i] > critical_mdot else Mdot_Rmag[i]
                                # add the new matter to decay the B field
                                Maccumulated = Mdot_NS[i] * deltaT[i] + Macc_accum # cgs
                                B[i + 1] = decay_law.decay_field(Maccumulated) # B[i-1]
                                Macc_accum += Mdot_NS[i] * deltaT[i] # cgs
                            # Rmag at Isco already, there's no B decay## and we assume the mass doesn't make it to the poles, 
                            # so Macc going towards the decay does not vary
                            else:
                                B[i + 1] = B[i]
                                Mdot_NS[i] = Mdot_Rmag[i]
                            
                        
                        mu = B[i] * neutron_star.R_NS**3 / 2
                        T_brake[i] = braking_torque(mu, neutron_star.Rlc)
                        T_mag[i] = magnetic_torque(B[i], Rmag, neutron_star)
                        T_spin_t[i] = T_disc[i] * cos(alpha_t[i]) + T_brake[i] * (1 + sin(chi_t[i])**2) + T_mag[i]
                        T_chi_t[i] = cos_function(eta, alpha_t[i], chi_t[i]) * T_disc[i] * sin(alpha_t[i])**2 * cos(alpha_t[i]) * sin(chi_t[i]) * cos(chi_t[i]) + T_brake[i] * sin(chi_t[i]) * cos(chi_t[i])
                        T_alpha_t[i] = -T_disc[i] * sin(alpha_t[i])
                        # update spin period
                        Pincr = deltaP(T_spin_t[i], deltaT[i], P_t[i], neutron_star.I)
                        P_t[i + 1] = P_t[i] + Pincr
                        # update alpha
                        alpha_incr = deltaangle(T_alpha_t[i], deltaT[i], neutron_star.omega, neutron_star.I)
                        alpha_t[i + 1] = alpha_t[i] + alpha_incr
                        # update chi
                        chi_incr = deltaangle(T_chi_t[i], deltaT[i], neutron_star.omega, neutron_star.I)
                        chi_t[i + 1] =  chi_t[i] + chi_incr

                except ValueError as e:
                    print("Math overflow. Aborting program", e)
                    B[i + 1:] = B[i]
                    P_t[i + 1:] = P_t[i]
                    alpha_t[i + 1:] = alpha_t[i]
                    chi_t[i + 1:] = chi_t[i]

                outdir = "field_suppression"
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)

                outdir = "%s/mdot_%.1f_P_%.3f_B_%.2E_%s_Chi_%.1f_a_%.1f" % (outdir,  mdot, P_init, B_init, decay_keyword, chi, alpha)

                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                
                outputs = np.array([B[:-1], P_t[:-1], Mdot_NS[:-1] * conversion, Mcrits[:-1] * conversion, Mdot_Rmag[:-1] * conversion, Rmags[:-1] / 1000, Rins[:-1] /1000, 
                                    Rsphs[:-1] / 1000, Rcor[:-1] / 1000, T_spin_t[:-1], alpha_t[:-1] / np.pi * 180, 
                                    chi_t[:-1] / np.pi * 180, propeller[:-1], pulsed[:-1], times[:-1]])
                np.savetxt("%s/results.dat" %(outdir), outputs.T, delimiter="\t",
                        fmt="%.7E\t%.5f\t%.4E\t%.4E\t%.4E\t%.3f\t%.3f\t%.3f\t%.3f\t%.2E\t%.2f\t%.2f\t%d\t%d\t%.5E", header="B\tP\tMacc\tMcrit\tMdotRmag\tRmag_km\tRin_km\tRsph_km\tRcor_km\ttorque_cm2_g_s2\talpha\tchi\tpropeller\tpulsed\tt")

                shutil.copy(args.config, outdir + "/" + os.path.basename(args.config))

                end = time.time()
                time_taken = end - start

                print("Done", "\n")

                print("Time taken: %.2f s" % time_taken)
                print(f"Final P:{P_t[-2]:.5f} s")
                print("Adding up mdot")
                M_acc = (np.cumsum(Mdot_NS[:-1] *  deltaT) * u.g).to(u.M_sun).value # convert to solar masses
                Mtotal = ((np.sum(Mdot_NS[:-1] *  deltaT) * u.g).to(u.M_sun)).value
                print("Total mass accreted: %.4f Msun" % Mtotal)

                fig, axes = plt.subplots(6, 1, sharex=True, gridspec_kw={"hspace":0.2}, figsize=(18, 14))
                i = 0
                ax = axes[i]
                scaling = 1
                plot_times = times[0:-1] / scaling
                ax.plot(plot_times, B[0:-1])
                #ax.fill_between(times[:-1] / scaling, 0, 1, where=propeller[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
                #ax.fill_between(times[:-1] / scaling, 0, 1, where=~pulsed[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
                ax.set_yscale("log")
                ax.margins(y=0.025)
                ax.set_xscale("log")
                ax.set_ylabel("B (G)")

                ax2 = ax.twinx()
                ax = ax2#axes[1]
                color = "C1"
                ax.plot(plot_times, P_t[0:-1], color=color, ls="--", lw=2.5)
                ax.tick_params("y", colors=color, which="both")
                ax.spines["right"].set_color(color)
                ax.yaxis.label.set_color(color)
                ax.set_ylabel("P(s)")
                ax.set_yscale("log")
                ax.set_xscale("log")

                if True:
                    ngc300ulx_data = np.genfromtxt("/home/andresgur/Documents/papers/magnetic_suppression/NGC300ULX1.dat", names=True, delimiter="\t")
                    # 2010 was discovered, 2014 first period
                    years = 4.5 # Nov 2014 - 2010
                    ax.errorbar((ngc300ulx_data["MJD"] - ngc300ulx_data["MJD"][0])/365 + years, ngc300ulx_data["P"], yerr=ngc300ulx_data["P_err"], 
                                label="NGC 300 ULX1", color=color, ls="None", fmt=".", markersize=20)
                    ax.legend()


                i+=1
                ax = axes[i]
                ax.plot(plot_times, Mdot[0:-1] * conversion * 10**6, label=r"$\dot{M}_0$", ls="--", color="black", alpha=0.9, lw=1.4)
                ax.plot(plot_times, Mdot_Rmag[0:-1] * conversion * 10**6, label=r"$\dot{M}(R_\mathrm{mag})$", ls=":")
                ax.scatter(plot_times, Mdot_NS[0:-1] * conversion * 10**6, label=r"$\dot{M}_\mathrm{NS}$")
                ax.plot(plot_times, Mcrits[0:-1] * conversion * 10**6, label=r"$\dot{M}_\mathrm{crit}$", ls="--", lw=2.5, color="C1")
                ax.legend()
                ax.set_ylabel('$\dot{M}$ (10$^{-6}$ $M_\odot$/yr)')
                ax.set_xscale("log")

                i+=1
                ax = axes[i]
                ax.plot(plot_times, M_acc[0:])
                ax.set_ylabel(r'$M_\mathrm{a}$ ($M_\odot$)')
                ax.set_xscale("log")
                ax.set_yscale("log")

                i+=1
                ax = axes[i]
                ax.scatter(plot_times, Rmags[0:-1] / 1000, label=r"$R_\mathrm{mag}$", color=np.where(pulsed[0:-1], "green", "red"), s=5)
                ax.plot(plot_times, Rsphs[0:-1]  / 1000, label=r"$R_\mathrm{sph}$", ls="--", lw=2.5)
                ax.plot(plot_times, Rcor[0:-1]  / 1000, label=r"$R_\mathrm{cor}$", ls=":", lw=3.5)
                #ax.plot(plot_times, Rins[0:-1]/1000, label=r"$R_\mathrm{in}$", color="black", ls="--", alpha=0.9)
                ax.legend()
                ax.set_ylabel('R (km)')
                ax.set_xscale("log")
                ax.set_yscale("log")

                i+=1
                ax = axes[i]
                ax.plot(plot_times, alpha_t[0:-1] / np.pi * 180, label=r"$\alpha$")
                ax.plot(plot_times, chi_t[0:-1] / np.pi * 180, label=r"$\chi$", ls="--")
                ax.legend()
                ax.set_ylabel(r'Angle ($^\circ$)')
                ax.set_xscale("log")

                i+=1
                ax = axes[i]
                N = (Mdot* np.sqrt(Gcgs * neutron_star.M * Rmags))[0:-1]
                ax.plot(plot_times, T_disc[0:-1] / N, label=r"$N_\mathrm{acc}$", lw=3)
                ax.plot(plot_times, T_mag[0:-1] / N, label=r"$N_\mathrm{mag}$", ls="--", lw=2.5)
                ax.plot(plot_times, T_brake[0:-1] / N, label=r"$N_\mathrm{psr}$",  ls=":")
                ax.axhline(ls="--", lw=2, color="black", zorder=-10)
                ax.legend()
                ax.set_ylabel(r'$N / \dot{M}_0 \sqrt{GMR_\mathrm{mag}}$')
                ax.set_xscale("log")


                axes[-1].set_xlabel("Time (yr)" )
                plt.savefig("%s/multiplot.png" % (outdir))
                plt.close(fig)

                fig = plt.figure()
                nudot = np.diff(1 / P_t) / deltaT
                plt.scatter(times[1:-1] / scaling, nudot[1:]) # convert to km
                plt.ylabel(r"$\dot{\nu}$ (Hz/s)")
                plt.xlabel("Time (yr)")
                plt.xscale("log")
                plt.yscale("log")
                ax = plt.gca()
                #ax.get_yaxis().get_major_formatter().set_useOffset(False)
                plt.savefig("%s/dpdt.png" % (outdir), dpi=100)
                plt.close(fig)

                fig = plt.figure()
                plt.scatter(plot_times, Rins[:-1] / 1000, ) # convert to km
                plt.scatter(plot_times, Rmags[:-1] / 1000, color=np.where(pulsed[0:-1], "green", "red"), s=5) # convert to km
                plt.axhline(neutron_star.R_NS / 1000, label=r"$R_\mathrm{NS}$", color="black", ls="--", alpha=0.9)
                plt.legend()
                plt.ylabel(r"$R_{isco}$ (km)")
                plt.xlabel("Time (yr)")
                plt.xscale("log")
                ax = plt.gca()
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                plt.yscale("log")
                plt.savefig("%s/isco.png" % (outdir), dpi=100)
                plt.close(fig)
                #plt.show()
                #plt.close(fig)
                #

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
