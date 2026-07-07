from field_decay.compact_objects import NS
from field_decay.constants import Gcgs, M_suncgs
from field_decay.accretion import (
    spherization_radius,
    spherization_radius_poutanen,
    magnetospheric_radius,
    magnetospheric_radius_superEdd,
    mass_transfer_rate_mag_radius_secant,
    mass_transfer_inner_radius,
    luminosity_super_edd_NS,
    neutron_star_binding_luminosity,
    fastness_parameter,
    mcrit,
    rmag_inclination_factor,
)

from field_decay.torques import (
    magnetic_torque_dai_propeller,
    accretion_torque_dai,
    accretion_torque,
    magnetic_torque_dai,
    magnetic_torque_radial_twisting,
)

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import argparse, time, os
from math import pi
from tqdm import tqdm
import shutil


def runmodel(
    neutron_star,
    mdots,
    rsph,
    e_wind,
    steps,
    deltaT_arr,
    gamma,
    delta,
    psi,
    Rmagfactor=1,
):
    nsteps = len(steps)
    # variables to be filled
    B_arr = [neutron_star.B] * nsteps  # G
    P_t = np.full(nsteps, neutron_star.P, dtype=float)
    Mdot_NS_arr = np.empty(nsteps, dtype=float)
    Mdot_Rmags = np.empty(nsteps, dtype=float)
    Mcrits_arr = np.zeros(nsteps, dtype=float)
    propeller = nsteps * [0]
    Rins = np.full(nsteps, neutron_star.Risco, dtype=float)
    Rmags = np.empty(nsteps, dtype=float)
    Rsphs = np.empty(nsteps, dtype=float)
    Rcor_arr = np.full(nsteps, neutron_star.Rco, dtype=float)
    alpha_t = np.full(nsteps, neutron_star.alpha, dtype=float)
    chi_t = np.full(nsteps, neutron_star.chi, dtype=float)
    T_disc_arr = np.empty(nsteps, dtype=float)
    T_mag_arr = np.empty(nsteps, dtype=float)
    T_twist_arr = np.empty(nsteps, dtype=float)
    T_brake_arr = np.empty(nsteps, dtype=float)
    luminosities_NS = np.zeros(nsteps, dtype=float)
    luminosities_disc = np.zeros(nsteps, dtype=float)
    luminosities_plunging = np.zeros(nsteps, dtype=float)

    Ledd = neutron_star.LEdd
    # mdot transferred is constant, even if MdotEdd increases, g/s
    Mdots = mdots * neutron_star.MEdd
    print("Mdot: %.1e g/s (%.1e M_sun/yr)" % (Mdots[0], Mdots[0] * conversion))

    M_NS = neutron_star.M
    R_NS = neutron_star.R_NS

    # for i, t in enumerate(times[1:], 1):
    for i in tqdm(steps[:-1]):
        Mdot_i = Mdots[i]
        deltaT = deltaT_arr[i]
        mdot = mdots[i]
        Mdot_isco = mass_transfer_inner_radius(mdot, e_wind) * Mdot_i
        B = B_arr[i]
        mu = neutron_star.mu
        # these change with the period
        Risco = neutron_star.Risco
        Rcor = neutron_star.Rco
        Rin = Risco if Risco > R_NS else R_NS  # cm
        # this is the NS radius or Rin (see Lipunova 1999 how the inner torque is defined)
        Rsph = rsph * Rin
        # store values
        Rsphs[i] = Rsph
        Rcor_arr[i] = Rcor
        Rmag = magnetospheric_radius(Mdot_i, mu, M_NS, psi=psi) * Rmagfactor

        # SS73 supercritical regime (thick disc)
        if Rmag < Rsph:
            # Rmag = magnetospheric_radius_wang_superEdd(Mdot[i-1], B[i-1], NS)#4.2 * 10**7 * (B[i-1] / 10**12) ** (4/9) # cm
            if e_wind == 0:
                # here we can calculate Rmag first thing and check
                Rmag = (
                    magnetospheric_radius_superEdd(neutron_star, psi=psi) * Rmagfactor
                )
                # if Rmag is larger than Rin the disc is truncated at Rmag
                if Rmag > Rin:
                    Rin = Rmag
                # this is the mass transfer rate at Rin, which is either Mdot(Rmag) if Rmag > Rin, or RNS or RIsco (whichever is larger)
                # if Rin > Rmag, then Rin cancel out above and below because Rsph = rsph x Rin
                Mdot_Rmag = Mdot_i * Rin / Rsph
            else:
                Mdot_Rmag = mass_transfer_rate_mag_radius_secant(
                    Mdot_i, mu, Mdot_isco, Rsph, M_NS, psi=psi, tol=1e-1, max_iter=500
                )
                Rmag = magnetospheric_radius(Mdot_Rmag, mu, M_NS, psi=psi) * Rmagfactor

                if Rmag > Rin:
                    Rin = Rmag
                # if Rmag drops below Rin then just assume Mdot Isco
                else:
                    Mdot_Rmag = Mdot_isco

            # if Rmag > Rin, then the disk is truncated at Rmag, otherwise at RNS or Risco whichever is larger
            luminosities_disc[i] = Ledd * (
                1 + luminosity_super_edd_NS(Rin, Rsph, e_wind)
            )

        # "subcritical" (thin disc) accretion
        else:
            Mdot_Rmag = Mdot_i
            if Rmag > Rin:
                Rin = Rmag
            # here Rin is either RNS or Risco, or Rmag if Rmag > than any of those two
            # this is the luminosity from the disc, which is truncated at Rmag or at Risco or RNS
            luminosities_disc[i] = Gcgs * M_NS * Mdot_Rmag / (2.0 * Rin)

        fastness = fastness_parameter(Rmag, Rcor)
        # the torque does not operate if Rmag is already at Rin, so we can set it to 0, otherwise we might get some numerical issues with the torque calculation

        # this is the torque due to the shearing of the radial component from Wang 1997
        # it's only non zero when the magnetic axis is misaligned (it is cancel later on at the torque function)
        # it should operate in both propeller and non propeller. It only operates when chi !=0, so for now we set it to 0
        T_twist = 0
        # Note if Rmag < RNS the accretion torque does not really makes much sense, neither the fastness parameter
        T_disc_arr[i] = accretion_torque_dai(
            Mdot_Rmag,
            Rmag,
            fastness,
            M_NS,
            delta=delta,
            gamma=gamma,
            psi=psi,
        )  # this torque works for both accretion and propeller
        # propeller (remember torque = 0 is spin eq, not propeller)
        if Rcor < Rmag:
            propeller[i] = 1
            B_arr[i + 1] = B
            magnetic_torque = magnetic_torque_dai_propeller
            Mdot_NS = 0.0
            # no contribution to L from the NS
        # accretion
        else:
            magnetic_torque = magnetic_torque_dai
            # accretion with magnetosphere --> B decays
            if Rmag > R_NS:
                Mcrit = mcrit(B)  # cgs
                # if we have exceeded the critical value, readjust for magnetic field suppression
                Mdot_NS = (
                    Mcrit if Mdot_Rmag > Mcrit else Mdot_Rmag
                )  # this is faster than min(X,X)
                # add the new matter to decay the B field
                neutron_star.decay_field(Mdot_NS * deltaT, Rmag)
                Mcrits_arr[i] = Mcrit
                # binding energy between Rmag and RNS
                luminosities_NS[i] = neutron_star_binding_luminosity(
                    Mdot_NS, Rmag, M_NS, R_NS
                )
                # If Rmag < Risco but > R_NS
                if Risco > Rmag:
                    # contribution from the plunging region
                    # here Rmag will be Risco or RNS, if RNS then this contribution will be zero
                    luminosities_plunging[i] = neutron_star_binding_luminosity(
                        Mdot_Rmag,
                        Risco,
                        M_NS,
                        Rmag,
                        beaming=1,
                    )
            # Rmag at RNS already, there's no B decay#
            # and we assume the mass doesn't make it to the poles,
            # so Macc going towards the decay does not vary
            else:
                Mdot_NS = Mdot_Rmag
                if Risco > R_NS:
                    luminosities_plunging[i] = neutron_star_binding_luminosity(
                        Mdot_Rmag,
                        Risco,
                        M_NS,
                        R_NS,
                        beaming=1,
                    )
        # if Rmag == RNs this is 0
        T_brake = neutron_star.braking_torque(Rmag)
        T_brake_arr[i] = T_brake
        Tmag = magnetic_torque(mu, Rmag, fastness, gamma=gamma, chi=chi_t[i])
        T_mag_arr[i] = Tmag
        T_twist_arr[i] = T_twist
        # update spin period, and alpha 
        neutron_star.torque(T_disc_arr[i], Tmag, T_twist, T_brake, deltaT)
        P_t[i + 1] = neutron_star.P
        # update alpha
        alpha_t[i + 1] = neutron_star.alpha
        # update chi
        chi_t[i + 1] = neutron_star.chi
        # store the magnetic field, which might have been updated if accretion occurd
        B_arr[i + 1] = neutron_star.B
        # store values that change depending on the disc config
        Rins[i] = Rin
        Rmags[i] = Rmag
        Mdot_Rmags[i] = Mdot_Rmag
        Mdot_NS_arr[i] = Mdot_NS

    return (
        P_t,
        B_arr,
        Mdots,
        Mdot_NS_arr,
        Mcrits_arr,
        Mdot_Rmags,
        Rmags,
        Rins,
        Rsphs,
        Rcor_arr,
        alpha_t,
        chi_t,
        T_disc_arr,
        T_mag_arr,
        T_twist_arr,
        T_brake_arr,
        luminosities_disc,
        luminosities_plunging,
        luminosities_NS,
        propeller,
    )


def read_config(file):
    config = {}
    with open(file, "r") as file:
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
                if key=="laws":
                    config[key] = values  # Strings for the decay laws
                else:
                    raise ValueError(f"Invalid value for key '{key}': {values}")

    return config


conversion = ((1 * (u.g / u.s)).to(u.Msun / u.yr)).value

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compute decaying field and period due to mass accreton rate in super-critical regime"
    )
    ap.add_argument("--config", nargs="?", help="Path to config file", required=True)
    ap.add_argument(
        "-o",
        "--outdir",
        nargs="?",
        help="Output directory",
        default="runs",
    )
    ap.add_argument(
        "--noplot",
        action="store_false",
        help="Whether to store the plots. Flag to deactivate, so it runs faster",
    )
    ap.add_argument(
        "--braking-model",
        choices=["Dipole", "EnhancedDipole"],
        default="Dipole",
        help="Select the braking prescription. Use 'enhanced dipole' to append the '_enhanced_dipole' suffix to the output directory.",
    )
    args = ap.parse_args()

    home = os.path.expanduser("~")
    stylefile = "%s/.config/matplotlib/stylelib/paper.mplstyle" % home
    if os.path.isfile(stylefile):
        plt.style.use("%s/.config/matplotlib/stylelib/paper.mplstyle" % home)

    parameters = read_config(args.config)

    B_inits = parameters["B"]
    P_inits = parameters["P"]
    decay_laws = parameters["laws"]
    input_mdots = parameters["mdot"]
    M_NS_sun = parameters["M"][0]
    R_NS = parameters["R"][0]
    # for now we do not vary this parameter
    chi = 0 # parameters["chi"][0]
    chirad = chi / 180 * np.pi
    Rmagfactor = rmag_inclination_factor(chirad)
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

    print(
        "Magnetospheric parameters:\n-------------------------\n gamma:%.2f\n delta:%.2f\n psi:%.2f\n"
        % (gamma, delta, psi)
    )
    # steps
    nsteps = int(parameters["steps"][0])
    tmax = parameters["tmax"][0]

    # deltaT = 0.5 * u.yr switched to log scale

    # times = np.arange(0, tmax, deltaT.to(u.yr).value) * u.yr
    # nsteps = 100000 #// 4# 50000
    print("Number of steps: %d" % nsteps)
    # times = np.concatenate((np.geomspace(0.1, tswitch, nsteps), np.linspace(tswitch, tmax, 2 * nsteps))) # in years
    times = np.geomspace(0.1, tmax, nsteps)  # # in years
    # times = np.arange(0.1, tmax, )
    steps = np.arange(nsteps)

    deltaT_arr = (np.diff(times) << u.yr).to(u.s).value
    print(f"Delta T early (days): {deltaT_arr[0] / 3600 / 24:.2f}")
    print(f"Delta T late (days): {deltaT_arr[-1] / 3600 / 24:.1f}")
    for mdot in input_mdots:
        for P_init in P_inits:
            for B_init in B_inits:
                for decay_law in decay_laws:

                    print(
                        "Running for mdot =  %.5f, P = %.3f s, B = %.2E, spin-axis angle: %.1f"
                        ", magnetic angle: %.1f, Tmax = %.1f yr decay law: %s"
                        % (mdot, P_init, B_init, alpha, chi, tmax, decay_law)
                    )

                    neutron_star = NS(
                        P=P_init,
                        M_NS=M_NS_sun,
                        R_NS=R_NS,
                        chi=chirad,
                        alpha=alpha / 180 * np.pi,
                        eta=eta,
                        B=B_init,
                        decay_law=decay_law,
                        braking_torque=args.braking_model,
                    )
                    print(neutron_star)
                    # print(f"Delta T (s):{deltaT:.2f}")
                    mdots = np.full(nsteps, mdot, dtype=float)

                    rsph = spherization_radius_formula(mdot, e_wind)
                    start = time.time()
                    (
                        P_t,
                        B_arr,
                        Mdots,
                        Mdot_NS_arr,
                        Mcrits_arr,
                        Mdot_Rmags,
                        Rmags,
                        Rins,
                        Rsphs,
                        Rcor_arr,
                        alpha_t,
                        chi_t,
                        T_disc_arr,
                        T_mag_arr,
                        T_twist_arr,
                        T_brake_arr,
                        luminosities_disc,
                        luminosities_plunging,
                        luminosities_NS,
                        propeller,
                    ) = runmodel(
                        neutron_star,
                        mdots,
                        rsph,
                        e_wind,
                        steps,
                        deltaT_arr,
                        gamma,
                        delta,
                        psi,
                    )

                    end = time.time()
                    time_taken = end - start

                    print("Done", "\n")

                    print("Time taken: %.2f s" % time_taken)
                    print(f"Final P:{P_t[-2]:.5f} s")
                    print("Adding up mdot")
                    M_acc = (
                        (np.cumsum(Mdot_NS_arr[:-1] * deltaT_arr) * u.g)
                        .to(u.M_sun)
                        .value
                    )  # convert to solar masses
                    Mtotal = (
                        (np.sum(Mdot_NS_arr[:-1] * deltaT_arr) * u.g).to(u.M_sun)
                    ).value
                    print("Total mass accreted: %.4f Msun" % Mtotal)

                    print(
                        "The source entered propeller %d times"
                        % np.count_nonzero(propeller)
                    )

                    outdir = args.outdir
                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)

                    out_suffix = ""
                    if args.braking_model == "EnhancedDipole":
                        out_suffix = "_enhanced_dipole"

                    outdir = (
                        "%s/mdot_%.1f_P_%.3f_B_%.2E_%s_Chi_%.1f_a_%.1f_ewind_%.1f_eta_%.2f%s"
                        % (
                            outdir,
                            mdot,
                            P_init,
                            B_init,
                            decay_law,
                            chi,
                            alpha,
                            e_wind,
                            eta,
                            out_suffix,
                        )
                    )

                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)

                    outputs = np.array(
                        [
                            B_arr[:-1],
                            P_t[:-1],
                            Mdots[:-1] * conversion,
                            Mdot_NS_arr[:-1] * conversion,
                            Mcrits_arr[:-1] * conversion,
                            Mdot_Rmags[:-1] * conversion,
                            Rmags[:-1] / 1000,
                            Rins[:-1] / 1000,
                            Rsphs[:-1] / 1000,
                            Rcor_arr[:-1] / 1000,
                            alpha_t[:-1] / pi * 180,
                            chi_t[:-1] / pi * 180,
                            luminosities_disc[:-1] / 10**39,
                            luminosities_plunging[:-1] / 10**39,
                            luminosities_NS[:-1] / 10**39,
                            propeller[:-1],
                            times[:-1],
                        ]
                    )

                    np.savetxt(
                        "%s/results.dat" % (outdir),
                        outputs.T,
                        delimiter="\t",
                        fmt="%.7E\t%.6f\t%.4E\t%.4E\t%.4E\t%.4E\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.0f\t%.5E",
                        header="B\tP\tMdot_0\tMacc\tMcrit\tMdotRmag\tRmag_km\tRin_km\tRsph_km\tRcor_km\talpha\tchi\tL_disc\tL_plung\tL_NS\tpropeller\tt",
                    )

                    shutil.copy(
                        args.config, outdir + "/" + os.path.basename(args.config)
                    )


                    if args.noplot:

                        fig, axes = plt.subplots(
                            6,
                            1,
                            sharex=True,
                            gridspec_kw={"hspace": 0.2},
                            figsize=(18, 14),
                        )
                        i = 0
                        ax = axes[i]
                        scaling = 1
                        plot_times = times[0:-1] / scaling
                        ax.plot(plot_times, B_arr[0:-1])
                        # ax.fill_between(times[:-1] / scaling, 0, 1, where=propeller[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
                        # ax.fill_between(times[:-1] / scaling, 0, 1, where=~pulsed[:-1], alpha=0.4, transform=ax.get_xaxis_transform(), color="red")
                        ax.set_yscale("log")
                        ax.margins(y=0.025)
                        ax.set_xscale("log")
                        ax.set_ylabel("B (G)")

                        ax2 = ax.twinx()
                        ax = ax2  # axes[1]
                        color = "C1"
                        ax.plot(plot_times, P_t[0:-1], color=color, ls="--", lw=2.5)
                        ax.tick_params("y", colors=color, which="both")
                        ax.spines["right"].set_color(color)
                        ax.yaxis.label.set_color(color)
                        ax.set_ylabel("P(s)")
                        ax.set_yscale("log")
                        ax.set_xscale("log")

                        i += 1
                        ax = axes[i]
                        ax.plot(
                            plot_times,
                            Mdots[0:-1] * conversion * 10**6,
                            label=r"$\dot{M}_0$",
                            ls="--",
                            color="black",
                            alpha=0.9,
                            lw=1.4,
                        )
                        ax.plot(
                            plot_times,
                            Mdot_Rmags[0:-1] * conversion * 10**6,
                            label=r"$\dot{M}(R_\mathrm{mag})$",
                            ls=":",
                            color="C2",
                        )
                        ax.scatter(
                            plot_times,
                            Mdot_NS_arr[0:-1] * conversion * 10**6,
                            label=r"$\dot{M}_\mathrm{NS}$",
                            color="C3",
                        )
                        ax.plot(
                            plot_times,
                            Mcrits_arr[0:-1] * conversion * 10**6,
                            label=r"$\dot{M}_\mathrm{crit}$",
                            ls="--",
                            lw=2.5,
                            color="C1",
                        )
                        ax.legend()
                        ax.set_ylabel(r"$\dot{M}$ (10$^{-6}$ $M_\odot$/yr)")
                        ax.set_xscale("log")

                        i += 1
                        ax = axes[i]
                        ax.plot(plot_times, M_acc[0:])
                        ax.set_ylabel(r"$M_\mathrm{a}$ ($M_\odot$)")
                        ax.set_xscale("log")
                        ax.set_yscale("log")

                        i += 1
                        ax = axes[i]
                        ax.scatter(
                            plot_times,
                            Rmags[0:-1] / 1000,
                            label=r"$R_\mathrm{mag}$",
                            color=np.where((Rmags > Rins)[0:-1], "green", "red"),
                            s=5,
                        )
                        ax.plot(
                            plot_times,
                            Rsphs[0:-1] / 1000,
                            label=r"$R_\mathrm{sph}$",
                            ls="--",
                            lw=2.5,
                        )
                        ax.plot(
                            plot_times,
                            Rcor_arr[0:-1] / 1000,
                            label=r"$R_\mathrm{cor}$",
                            ls=":",
                            lw=3.5,
                        )
                        # ax.plot(plot_times, Rins[0:-1]/1000, label=r"$R_\mathrm{in}$", color="black", ls="--", alpha=0.9)
                        ax.legend()
                        ax.set_ylabel("R (km)")
                        ax.set_xscale("log")
                        ax.set_yscale("log")

                        i += 1
                        ax = axes[i]
                        ax.plot(
                            plot_times, alpha_t[0:-1] / np.pi * 180, label=r"$\alpha$"
                        )
                        ax.plot(
                            plot_times,
                            chi_t[0:-1] / np.pi * 180,
                            label=r"$\chi$",
                            ls="--",
                        )
                        ax.legend()
                        ax.set_ylabel(r"Angle ($^\circ$)")
                        ax.set_xscale("log")

                        i += 1
                        ax = axes[i]
                        N = accretion_torque(Mdots, M_NS_sun * M_suncgs, Rmags)[0:-1]
                        ax.plot(
                            plot_times,
                            T_disc_arr[0:-1] / N,
                            label=r"$N_\mathrm{acc}$",
                            lw=3,
                        )
                        ax.plot(
                            plot_times,
                            T_mag_arr[0:-1] / N,
                            label=r"$N_\mathrm{mag}$",
                            ls="--",
                            lw=2.5,
                        )
                        ax.plot(
                            plot_times,
                            T_twist_arr[0:-1] / N,
                            label=r"$N_\mathrm{r}$",
                            ls="--",
                            lw=2.5,
                        )
                        ax.plot(
                            plot_times,
                            T_brake_arr[0:-1] / N,
                            label=r"$N_\mathrm{psr}$",
                            ls=":",
                        )
                        ax.axhline(ls="--", lw=2, color="black", zorder=-10)
                        ax.legend()
                        ax.set_ylabel(r"$N / \dot{M}_0 \sqrt{GMR_\mathrm{mag}}$")
                        ax.set_xscale("log")

                        axes[-1].set_xlabel("Time (yr)")
                        plt.savefig("%s/multiplot.png" % (outdir))
                        plt.close(fig)

                        fig = plt.figure()
                        # nudot = np.diff(1 / P_t) / deltaT_arr
                        yr_in_seconds = 365.25 * 24.0 * 3600.0
                        nup = np.diff(P_t) / deltaT_arr  # in s/s
                        plt.scatter(
                            plot_times[1:] / scaling, -nup[1:] / 10**-11
                        )  # convert to km
                        nup = np.gradient(
                            P_t, times * yr_in_seconds
                        )  # convert yr to seconds
                        plt.plot(
                            plot_times[1:] / scaling, -nup[1:-1] / 10**-11, color="C1"
                        )  # convert to km
    
                        plt.legend()
                        # plt.ylabel(r"$\dot{\nu}$ (Hz/s)")
                        plt.ylabel(r"$-\dot{P}$ ($10^{-11}$ s/s)")
                        plt.xlabel("Time (yr)")
                        plt.xscale("log")
                        plt.yscale("log")
                        ax = plt.gca()
                        # twin y axis showing yr/s
                        ax2 = ax.twinx()
                        ax2.set_yscale("log")
                        ylims = ax.get_ylim()
                        ax2.set_ylim(
                            ylims[0] * 10**-11 * yr_in_seconds,
                            ylims[1] * 10**-11 * yr_in_seconds,
                        )
                        ax2.set_ylabel(r"$-\dot{P}$ (yr/s)")
                        # ax.get_yaxis().get_major_formatter().set_useOffset(False)
                        plt.savefig("%s/dpdt.png" % (outdir), dpi=100)
                        plt.close(fig)

                        fig = plt.figure()
                        plt.scatter(
                            plot_times, Rins[:-1] / 1000, label=r"$R_\mathrm{in}$"
                        )  # convert to km
                        plt.scatter(
                            plot_times, Rsphs[:-1] / 1000, label=r"$R_\mathrm{sph}$"
                        )  # convert to km
                        plt.scatter(
                            plot_times,
                            Rmags[:-1] / 1000,
                            color=np.where((Rmags > Rins)[0:-1], "green", "red"),
                            s=5,
                            label=r"$R_\mathrm{mag}$",
                        )  # convert to km
                        plt.axhline(
                            R_NS / 1000,
                            label=r"$R_\mathrm{NS}$",
                            color="black",
                            ls="--",
                            alpha=0.9,
                        )
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
                        plt.plot(
                            plot_times,
                            luminosities_NS[:-1] / 10**39,
                            label=r"$L_\mathrm{NS}$",
                            ls="--",
                        )
                        plt.plot(
                            plot_times,
                            luminosities_disc[:-1] / 10**39,
                            label=r"$L_\mathrm{disc}$",
                            ls=":",
                        )
                        plt.plot(
                            plot_times,
                            luminosities_plunging[:-1] / 10**39,
                            label=r"$L_\mathrm{plung}$",
                        )
                        plt.plot(
                            plot_times,
                            (
                                luminosities_plunging[:-1]
                                + luminosities_disc[:-1]
                                + luminosities_NS[:-1]
                            )
                            / 10**39,
                            color="black",
                            label=r"$L_\mathrm{total}$",
                            zorder=-10,
                        )
                        plt.legend()
                        plt.axhline(1, ls="--", color="black")  # convert to km
                        plt.ylabel(r"$L$ (10$^{39}$ erg/s)")
                        plt.xlabel("Time (yr)")
                        plt.xscale("log")
                        ax = plt.gca()
                        ax.get_yaxis().get_major_formatter().set_useOffset(False)
                        plt.savefig("%s/L.png" % (outdir), dpi=100)
                        plt.close(fig)
                        # plt.figure()
                        # plt.scatter(plot_times, pulsed[:-1])
                        # plt.xlim(4 *10**5, plot_times[-1])
                        # plt.ylabel("Pulsed")
                        # plt.xscale("log")
                        # plt.savefig("%s/pulsed.png" % outdir, dpi=100)
                        # plt.show()
                        # plt.close(fig)
                        #

                        print("Stored plots")
                    else:
                        print("Plots won't be stored")
                    print("Outputs saved  to %s" % outdir)
