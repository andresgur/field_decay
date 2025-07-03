from compact_object import NS, Gcgs
from accretion import *
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import argparse, warnings, time, os
from math import log, pi
from celerite.terms import RealTerm
from celerite import GP
from tqdm import tqdm
from field_decay_law import ShibazakiFieldDecay, PayneFieldDecay, ZhangFieldDecayDiff
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
                config[key] = values # Strings for the decay laws

    return config

conversion = ((1 * (u.g/u.s)).to(u.Msun / u.yr)).value
DECAY_LAWS = ["Payne", "Shibazaki", "Zhang"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Compute decaying field and period due to mass accreton rate in super-critical regime')
    ap.add_argument("--config", nargs="?", help="Path to config file", required=True)
    ap.add_argument("-o", "--outdir", nargs="?", help="Output directory", default="field_suppression")
    args = ap.parse_args()

    home = os.getenv("HOME")
    stylefile = '%s/.config/matplotlib/stylelib/paper.mplstyle' % home
    if os.path.isfile(stylefile):
        plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)
    
    parameters = read_config(args.config)

    B_inits = parameters["B"]
    P_inits = parameters["P"]
    decay_laws = parameters["laws"]
    input_mdots = parameters["mdot"]
    M_NS = parameters["M"][0] * u.M_sun
    R_NS = parameters["R"][0]
    chi = parameters["chi"][0]
    alpha = parameters["alpha"][0]
    eta = parameters["eta"][0]
    # magnetosphere
    gamma = parameters["gamma"][0]
    delta = parameters["delta"][0]
    psi = parameters["psi"][0]
    e_wind = parameters["e_wind"][0]

    if e_wind == 0:
        print("Advection will be ignored")
        spherization_radius_formula = spherization_radius
    else:
        print("Advection with e_wind = %.2f" % e_wind)
        spherization_radius_formula = spherization_radius_poutanen

    print("Magnetospheric parameters:\n-------------------------\n\gamma:%.2f\n\delta:%.2f\n\psi:%.2f\n" % (gamma, delta, psi))
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
    steps = np.arange(nsteps)
    
    deltaT = (np.diff(times) << u.yr).to(u.s).value
    print(f"Delta T early (days): {deltaT[0] / 3600 / 24:.2f}")
    print(f"Delta T late (days): {deltaT[-1] / 3600 / 24:.1f}")
    for mdot in input_mdots:
        for P_init in P_inits:
            for B_init in B_inits:
                for decay_law in decay_laws:
                    if decay_law in DECAY_LAWS:
                        decay_keyword =  decay_law
                    else:
                        warnings.warn("Decay keyword %s not valid! Available options are: %s" % (decay_law, " ".join(DECAY_LAWS)))
                        continue

                    print("Running for mdot =  %.5f, P = %.3f s, B = %.2E, spin-axis angle: %.1f" \
                        ", magnetic angle: %.1f, Tmax = %.1f yr decay law: %s" % (mdot, P_init, B_init, alpha, chi, tmax, decay_keyword))

                    neutron_star = NS(P_NS = P_init, M_NS=M_NS, R_NS=R_NS, chi=chi / 180 * np.pi, alpha=alpha / 180 * np.pi, eta=eta)
                    #print(f"Delta T (s):{deltaT:.2f}")

                    # variables to be filled
                    B = [B_init] * nsteps # G
                    P_t = P_init * np.ones(nsteps)
                    Mdot_NS = np.empty(nsteps)
                    Mdot_Rmag = np.empty(nsteps)
                    Mcrits = np.zeros(nsteps)
                    propeller = [0] * nsteps
                    Rins = np.ones(nsteps) * neutron_star.Risco
                    Rmags = np.empty(nsteps)
                    Rsphs = np.empty(nsteps)
                    Rcor = np.ones(nsteps) * neutron_star.Rco
                    alpha_t = np.ones(nsteps) * neutron_star.alpha
                    chi_t = np.ones(nsteps) * neutron_star.chi
                    T_disc = np.empty(nsteps)
                    T_mag = np.empty(nsteps)
                    T_brake = np.empty(nsteps)
                    mdots = mdot  * np.ones(nsteps)
                    luminosities_NS = np.zeros(nsteps)
                    luminosities_disc = np.zeros(nsteps)
                    luminosities_plunging = np.zeros(nsteps)
                    # mean of mdot, 1 day bendtimescale, variance = 1 mdot
                    timescale = 10 / 365

                    Ledd = eddington_luminosity(M_NS.value)

                    np.random.seed(15)
                    Fvar = 0.20
                    variance = Fvar**2 * mdot**2
                    kernel = RealTerm(log_a=log(variance), log_c=log(2 * pi / timescale)) 
                    gp = GP(kernel, mean=mdot)
                    gp.compute(times)
                    #mdots = gp.sample()
                    if np.any(mdots < 0):
                        raise ValueError("There are %d negative mdots" % (np.count_nonzero(mdots < 0)))
                    #print(mdots[np.isnan(mdots)])
                    #mdots[np.isnan(mdots)] = mdot

                    Mdot = mdots * neutron_star.Medd # mdot transferred is constant, even if MdotEdd increases, g/s
                    print("Mdot: %.1e g/s (%.1e M_sun/yr)" % (Mdot[0], Mdot[0] * conversion))

                    if decay_keyword=="Zhang":
                        decay_law = ZhangFieldDecayDiff(B_init)
                    elif decay_keyword=="Shibazaki":
                        decay_law = ShibazakiFieldDecay(B_init)
                    elif decay_keyword == "Payne":
                        decay_law = PayneFieldDecay(B_init)
                        print("Value of n: %.2f" % decay_law.n)
                    print("Using decay law from: %s" % decay_law.name)
                    start = time.time()
                
                    #for i, t in enumerate(times[1:], 1):
                    for i in tqdm(steps[:-1]):
                        # update the period so that we get a new Rco and Risco as well as MdotEdd
                        Rins[i] = neutron_star.Risco if neutron_star.Risco > neutron_star.R_NS else neutron_star.R_NS # cm
                        Rcor[i] = neutron_star.Rco
                        Rsphs[i] = spherization_radius_formula(mdots[i], Rins[i], e_wind=e_wind) # cm this is the NS radius or Rin (see Lipunova 1999 how the inner torque is defined)
                        #Rmag, err = magnetospheric_radius_wang(Mdot[i-1], B[i-1], NS) # cm
                        mu = magnetic_moment(B[i], neutron_star.R_NS)
                        Rmag = magnetospheric_radius(Mdot[i], mu, neutron_star.M, psi=psi)
                        
                        # SS73 supercritical regime (thick disc)
                        if Rmag < Rsphs[i]:
                            #Rmag = magnetospheric_radius_wang_superEdd(Mdot[i-1], B[i-1], NS)#4.2 * 10**7 * (B[i-1] / 10**12) ** (4/9) # cm
                            if e_wind==0:
                                Rmag = magnetospheric_radius_superEdd(mu, neutron_star.M, psi=psi)
                                Mdot_Rmag[i] = Mdot[i] * Rmag / Rsphs[i]
                                if (Rmag <= Rins[i]):
                                    Mdot_Rmag[i] = Mdot[i] * Rmag / Rsphs[i]
                                    if Rmag <= neutron_star.R_NS:
                                        Rmag = neutron_star.R_NS 
                                else:
                                    Rins[i] = Rmag
                            else:
                                Mdot_Rmag[i] = mass_transfer_rate_mag_radius(Mdot[i], mu, neutron_star.Medd, Rsphs[i], neutron_star.M, psi=psi, e_wind=e_wind)
                                Rmag = magnetospheric_radius(Mdot_Rmag[i], mu, neutron_star.M, psi=psi)
                                
                                if (Rmag <= Rins[i]):
                                    Mdot_Rmag[i] = mass_transfer_inner_radius(mdots[i], e_wind) * Mdot[i]
                                    if (Rmag <= neutron_star.R_NS):
                                        Rmag = neutron_star.R_NS
                                else:
                                    Rins[i] = Rmag
                            # for numerical stability (mainly with Payne's mangetic field decay) let's allow some tolerance
                            #if Rins[i] - TOLERANCE < Rmag < Rins[i] + TOLERANCE: # non magnetic accretion, no B field decay
                            luminosities_disc[i] = Ledd * (1 + luminosity_super_edd_NS(Rins[i], Rsphs[i], e_wind))
                            
                        # "subcritical" (thin disc) accretion
                        else:
                            Mdot_Rmag[i] = Mdot[i]
                            luminosities_disc[i] = Gcgs * neutron_star.M * Mdot_Rmag[i] / (2 * Rmag)

                        Rmags[i] = Rmag
                        fastness = fastness_parameter(Rmags[i], Rcor[i])
                        T_disc[i] = accretion_torque_dai(Mdot_Rmag[i], Rmag, fastness, neutron_star.M, 
                                                         delta=delta, gamma=gamma, psi=psi) # this torque works for both accretion and propeller
                        # propeller (remember torque = 0 is spin eq, not propeller)
                        if Rcor[i] < Rmags[i]:
                            propeller[i] = 1
                            B[i + 1] = B[i]
                            magnetic_torque = magnetic_torque_dai_propeller
                            # no contribution to L from the NS
                        # accretion
                        else:
                            magnetic_torque = magnetic_torque_dai
                            # accretion with magnetosphere --> B decays
                            if Rmags[i] > neutron_star.R_NS:
                                critical_mdot = mcrit(B[i]) # cgs
                                Mcrits[i] = critical_mdot
                                # if we have exceeded the critical value, readjust for magnetic field suppression
                                Mdot_NS[i] = critical_mdot if Mdot_Rmag[i] > critical_mdot else Mdot_Rmag[i] # this is faster than min(X,X)
                                # add the new matter to decay the B field
                                decay_law.decay_field(Mdot_NS[i] * deltaT[i], Rmag)
                                B[i + 1] = decay_law.B
                                # If Rmag < Risco but > R_NS
                                if Rmags[i] < neutron_star.Risco:
                                    # contribution from the plunging region
                                    luminosities_plunging[i] = neutron_star_binding_luminosity(Mdot_Rmag[i], neutron_star.Risco, neutron_star.M, Rmag, beaming=1)
                            # Rmag at RNS already, there's no B decay#
                            # and we assume the mass doesn't make it to the poles, 
                            # so Macc going towards the decay does not vary
                            else:
                                B[i + 1] = B[i]
                                Mdot_NS[i] = Mdot_Rmag[i]                

                        # here Rmag will be Risco or RNS, if RNS then this contribution will be zero 
                        luminosities_NS[i] = neutron_star_binding_luminosity(Mdot_NS[i], Rmag, neutron_star.M, neutron_star.R_NS)
                        T_brake[i] = braking_torque(mu, neutron_star.Rlc)
                        T_mag[i] = magnetic_torque(mu, Rmag, fastness, gamma=gamma)
                        #T_spin_t[i] = T_disc[i] * cos(alpha_t[i]) + T_brake[i] * (1 + sin(chi_t[i])**2) + T_mag[i]
                        #T_alpha_t[i] = -T_disc[i] * sin(alpha_t[i])
                        # update spin period
                        #Pincr = deltaP(T_spin_t[i], deltaT[i], P_t[i], neutron_star.I)
                        try:
                            neutron_star.torque(T_disc[i], T_mag[i], T_brake[i], deltaT[i])
                        except ZeroDivisionError as e:
                            print("Error in torque calculation. Aborting program")
                            print(e)
                            break
                        P_t[i + 1] = neutron_star.P_NS
                        # update alpha
                        #alpha_incr = deltaangle(T_alpha_t[i], deltaT[i], neutron_star.omega, neutron_star.I)
                        alpha_t[i + 1] = neutron_star.alpha
                        # update chi
                        #chi_incr = deltaangle(T_chi_t[i], deltaT[i], neutron_star.omega, neutron_star.I)
                        chi_t[i + 1] = neutron_star.chi

                #except ValueError as e:
                 #       print("Math overflow. Aborting program", e)
                  #      B[i + 1:] = B[i]
                   #     P_t[i + 1:] = P_t[i]
                    #    alpha_t[i + 1:] = alpha_t[i]
                     #   chi_t[i + 1:] = chi_t[i]

                    outdir = args.outdir
                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)

                    outdir = "%s/mdot_%.1f_P_%.3f_B_%.2E_%s_Chi_%.1f_a_%.1f_ewind_%.1f_eta_%.2f" % (outdir,  mdot, P_init, B_init, decay_keyword, chi, alpha, e_wind, eta)

                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)
                    
                    outputs = np.array([B[:-1], P_t[:-1], Mdot[:-1] * conversion, Mdot_NS[:-1] * conversion, Mcrits[:-1] * conversion, Mdot_Rmag[:-1] * conversion, Rmags[:-1] / 1000, Rins[:-1] /1000, 
                                        Rsphs[:-1] / 1000, Rcor[:-1] / 1000, alpha_t[:-1] / np.pi * 180, 
                                        chi_t[:-1] / np.pi * 180, luminosities_disc[:-1] / 10**39, luminosities_plunging[:-1] / 10**39, luminosities_NS[:-1] / 10**39, propeller[:-1], times[:-1]])
                    np.savetxt("%s/results.dat" %(outdir), outputs.T, delimiter="\t",
                            fmt="%.7E\t%.6f\t%.4E\t%.4E\t%.4E\t%.4E\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.5E", header="B\tP\tMdot_0\tMacc\tMcrit\tMdotRmag\tRmag_km\tRin_km\tRsph_km\tRcor_km\talpha\tchi\tL_disc\tL_plung\tL_NS\tpropeller\tt")

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

                    print("The source entered propeller %d times" % np.count_nonzero(propeller))

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
                    ax.plot(plot_times, Mdot_Rmag[0:-1] * conversion * 10**6, label=r"$\dot{M}(R_\mathrm{mag})$", ls=":", color="C2")
                    ax.scatter(plot_times, Mdot_NS[0:-1] * conversion * 10**6, label=r"$\dot{M}_\mathrm{NS}$", color="C3")
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
                    ax.scatter(plot_times, Rmags[0:-1] / 1000, label=r"$R_\mathrm{mag}$", color=np.where((Rmags > Rins)[0:-1], "green", "red"), s=5)
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
                    #plt.yscale("log")
                    ax = plt.gca()
                    #ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    plt.savefig("%s/dpdt.png" % (outdir), dpi=100)
                    plt.close(fig)

                    fig = plt.figure()
                    plt.scatter(plot_times, Rins[:-1] / 1000, ) # convert to km
                    plt.scatter(plot_times, Rsphs[:-1] / 1000, label="$R_\mathrm{sph}$") # convert to km
                    plt.scatter(plot_times, Rmags[:-1] / 1000, color=np.where((Rmags>Rins)[0:-1], "green", "red"), s=5, label="$R_\mathrm{mag}$") # convert to km
                    plt.axhline(neutron_star.R_NS / 1000, label=r"$R_\mathrm{NS}$", color="black", ls="--", alpha=0.9)
                    plt.legend()
                    plt.ylabel(r"$R$ (km)")
                    plt.xlabel("Time (yr)")
                    plt.xscale("log")
                    ax = plt.gca()
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    plt.yscale("log")
                    plt.savefig("%s/radii.png" % (outdir), dpi=100)
                    plt.close(fig)


                    fig = plt.figure()
                    plt.plot(plot_times, luminosities_NS[:-1] / 10**39, label="$L_\mathrm{NS}$", ls="--")
                    plt.plot(plot_times, luminosities_disc[:-1] / 10**39, label="$L_\mathrm{disc}$", ls=":")
                    plt.plot(plot_times, luminosities_plunging[:-1] / 10**39, label="$L_\mathrm{plung}$")
                    plt.plot(plot_times, (luminosities_plunging[:-1] + luminosities_disc[:-1] + luminosities_NS[:-1]) / 10**39, 
                             color="black", label="$L_\mathrm{total}$", zorder=-10)
                    plt.legend()
                    plt.axhline(1, ls="--", color="black") # convert to km
                    plt.ylabel(r"$L$ (10$^{39}$ erg/s)")
                    plt.xlabel("Time (yr)")
                    plt.xscale("log")
                    ax = plt.gca()
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    plt.savefig("%s/L.png" % (outdir), dpi=100)
                    plt.close(fig)
                    #plt.figure()
                    #plt.scatter(plot_times, pulsed[:-1])
                    #plt.xlim(4 *10**5, plot_times[-1])
                    #plt.ylabel("Pulsed")
                    #plt.xscale("log")
                    #plt.savefig("%s/pulsed.png" % outdir, dpi=100)
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
                    print("Saved to %s" % outdir)
