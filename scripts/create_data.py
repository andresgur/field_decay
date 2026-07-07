from field_decay.compact_objects import NS
from field_decay.constants import Gcgs
from field_decay.accretion import spherization_radius, spherization_radius_poutanen
from field_suppression import read_config, runmodel
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import argparse, time, os
import shutil


def check_offset(value):
    ivalue = float(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f"Invalid value {value}. --offset must be positive"
        )
    return ivalue


conversion = ((1 * (u.g / u.s)).to(u.Msun / u.yr)).value
DECAY_LAWS = ["Payne", "Shibazaki", "Zhang"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compute decaying field and period due to mass accreton rate in super-critical regime"
    )
    ap.add_argument("--config", nargs="?", help="Path to config file", required=True)
    ap.add_argument(
        "-o",
        "--outdir",
        nargs="?",
        help="Output directory, Default mock_data",
        default="mock_data",
    )
    ap.add_argument(
        "--offset",
        nargs="?",
        help="Time offset in years (>). Default 0 yr",
        default=0,
        type=check_offset,
    )
    ap.add_argument(
        "-d",
        "--data",
        nargs=1,
        help="Data with the observing window and uncertainties",
        required=True,
    )

    args = ap.parse_args()

    home = os.getenv("HOME")
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
    times = np.geomspace(0.005, tmax, nsteps)  # # in years
    # times = np.arange(0.1, tmax, )
    steps = np.arange(nsteps)

    deltaT_arr = (np.diff(times) << u.yr).to(u.s).value
    print(f"Delta T early (days): {deltaT_arr[0] / 3600 / 24:.2f}")
    print(f"Delta T late (days): {deltaT_arr[-1] / 3600 / 24:.1f}")
    for mdot in input_mdots:
        rsph = spherization_radius_formula(mdot, e_wind)
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
                        chi=chi / 180 * np.pi,
                        alpha=alpha / 180 * np.pi,
                        eta=eta,
                        B=B_init,
                        decay_law=decay_law,
                    )
                    print(neutron_star)
                    # print(f"Delta T (s):{deltaT:.2f}")
                    mdots = np.full(nsteps, mdot, dtype=float)
                    # mean of mdot, 1 day bendtimescale, variance = 1 mdot

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
                        T_disk_arr,
                        T_twist_arr,
                        T_mag_arr,
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

                    outdir = args.outdir
                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)

                    outdir = (
                        "%s/mdot_%.1f_P_%.3f_B_%.2E_%s_Chi_%.1f_a_%.1f_ewind_%.1f_eta_%.2f_offset_%d"
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
                            args.offset,
                        )
                    )

                    if not os.path.isdir(outdir):
                        os.mkdir(outdir)

                    observed_data = np.genfromtxt(
                        "%s" % args.data[0], names=True, delimiter="\t"
                    )
                    print(f"Number of datapoints {len(observed_data)}")
                    observed_data = np.sort(observed_data, order="MJD")
                    times_data = (observed_data["MJD"] - observed_data["MJD"][0]) / 365

                    indexes = [
                        np.argmin(np.abs(times_data[i] - (times - args.offset)))
                        for i in range(len(times_data))
                    ]
                    Periods = P_t[indexes]
                    luminosities = (
                        luminosities_disc + luminosities_NS + luminosities_plunging
                    )[indexes]
                    Porb = 5
                    # shake the periods using the uncertainties and add some additional scatter due to the orbit
                    newperiods = (
                        np.random.normal(Periods, observed_data["P_err"])
                        + np.sin(Porb / 365) * 10**-9
                    )
                    # the luminosities probably have higher uncertainty
                    L_err_observed = observed_data["L_err"] * 10**38.0
                    newLs = np.random.normal(luminosities, 1.5 * L_err_observed)
                    print("WARNING: Not randomizing datapoints for now")
                    outputs = np.array(
                        [
                            observed_data["MJD"],
                            Periods,
                            observed_data["P_err"],
                            luminosities / 10**38,
                            L_err_observed / 10**38,
                        ]
                    )
                    np.savetxt(
                        "%s/mock_data.dat" % outdir,
                        outputs.T,
                        header="MJD\tP\tP_err\tL\tL_err",
                        fmt="%.5f\t%.5f\t%.6E\t%.2f\t%.2f",
                    )

                    print("Saved to %s" % outdir)

                    fig, axes = plt.subplots(
                        5, 1, sharex=True, gridspec_kw={"hspace": 0.2}, figsize=(18, 14)
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
                    # ax.set_xscale("log")
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
                    # ax.set_xscale("log")

                    ax.errorbar(
                        times_data + args.offset,
                        newperiods,
                        yerr=observed_data["P_err"],
                        label=r"Fake Data",
                        color=color,
                        ls="None",
                        fmt=".",
                        markersize=20,
                    )
                    ax.legend()

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
                    ax.set_yscale("log")

                    i += 1
                    ax = axes[i]
                    ax.plot(plot_times, alpha_t[0:-1] / np.pi * 180, label=r"$\alpha$")
                    ax.plot(
                        plot_times, chi_t[0:-1] / np.pi * 180, label=r"$\chi$", ls="--"
                    )
                    ax.legend()
                    ax.set_ylabel(r"Angle ($^\circ$)")

                    i += 1
                    ax = axes[i]
                    N = (Mdots * np.sqrt(Gcgs * neutron_star.M * Rmags))[0:-1]
                    ax.plot(
                        plot_times,
                        T_disk_arr[0:-1] / N,
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

                    # copy the config file
                    shutil.copy(
                        args.config, outdir + "/" + os.path.basename(args.config)
                    )
