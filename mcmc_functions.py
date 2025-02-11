from compact_object import NS
from accretion import accretion_torque_dai, magnetic_torque_dai, magnetic_torque_dai_propeller, mass_transfer_inner_radius, spherization_radius_poutanen, mass_transfer_rate_mag_radius, magnetospheric_radius, mcrit, magnetic_moment, fastness_parameter
import numpy as np
import astropy.units as u
from field_decay_law import ShibazakiFieldDecay
from field_suppression import braking_torque
import cython

def chi_square(params, times, times_data, P_data, P_err, deltaT, nsteps: cython.int, steps):

    # interpolate and do chi-sq
    P_t = model(params[1], params[2], params[3], deltaT, nsteps, steps)
    periods = np.interp(times_data, times + params[0], P_t, right=np.inf, left=np.inf)
    return np.sum( ((periods - P_data)/ P_err)**2)


def neg_chi_square(params, times, times_data, P_data, P_err, parambounds, deltaT, nsteps: cython.int, steps):

    if not np.all(np.logical_and(parambounds[:, 0] <= params, params <= parambounds[:, 1])):
        return -np.inf
    return -chi_square(params, times, times_data, P_data, P_err, deltaT, nsteps, steps)


def model(logMdot: cython.double, P_init: cython.double, log_B_init: cython.double, deltaT, nsteps: cython.int, steps, chi: cython.double=0, alpha: cython.double=0, 
          gamma: cython.float=1, delta: cython.float=0.1, 
          psi: cython.float=0.5, e_wind: cython.float=0.5, M_NS=1.4 * u.M_sun, R_NS: cython.double=10**6):

    neutron_star = NS(P_NS = P_init, M_NS=M_NS, R_NS=R_NS, chi=chi, alpha=alpha)
    B_init:cython.double  = 10.**log_B_init
    decay_law = ShibazakiFieldDecay(B_init)
    B = B_init * np.ones(nsteps) # G
    P_t = P_init * np.ones(nsteps)
    #Mdot_NS = np.zeros(nsteps)
    #Mdot_Rmag = np.zeros(nsteps)
    #pulsed = np.ones(nsteps)
    #Rins = np.ones(nsteps) * neutron_star.Risco
    #alpha_t = np.ones(nsteps) * neutron_star.alpha
    #chi_t = np.ones(nsteps) * neutron_star.chi
    Mdot:cython.double = 10.**logMdot
    mdot:cython.double = Mdot / neutron_star.Medd
    pulsed:cython.int
    pulsed = 1
    mu: cython.double
    Rmag: cython.double
    Rsph: cython.double
    fastness: cython.double
    T_disc: cython.double
    T_brake: cython.double
    T_mag: cython.double
    Mdot_Rmag: cython.double
    Mdot_NS: cython.double
    i: cython.int
    try:
        for i in steps[:-1]:
            
            Rin = neutron_star.Risco if neutron_star.Risco > neutron_star.R_NS else neutron_star.R_NS # cm
            Rsph = spherization_radius_poutanen(mdot, Rin, e_wind=e_wind)
            mu = magnetic_moment(B[i], neutron_star.R_NS)
            Rmag = magnetospheric_radius(Mdot, mu, neutron_star.M, psi=psi)
            # SS73 supercritical regime
            if Rmag < Rsph:
                Mdot_Rmag = mass_transfer_rate_mag_radius(Mdot, mu, neutron_star.Medd, Rsph, neutron_star.M, psi=psi, e_wind=e_wind)
                Rmag = magnetospheric_radius(Mdot_Rmag, mu, neutron_star.M, psi=psi)
                if Rmag <= Rin:
                    Rmag = Rin
                    pulsed = 0
                    Mdot_Rmag = mass_transfer_inner_radius(mdot, e_wind) * Mdot
            # "subcritical" accretion
            else:
                Mdot_Rmag = Mdot
            fastness = fastness_parameter(Rmag, neutron_star.Rco)
            T_disc = accretion_torque_dai(Mdot_Rmag, Rmag, fastness, neutron_star.M, 
                                        delta=delta, gamma=gamma, psi=psi)# this torque works for both accretion and propeller
            # propeller (remember torque = 0 is spin eq, not propeller)
            if neutron_star.Rco < Rmag:
                B[i + 1] = B[i]
                magnetic_torque = magnetic_torque_dai_propeller
            # accretion
            else:
                magnetic_torque = magnetic_torque_dai
                # accretion with magnetosphere --> B decays
                if pulsed:
                    critical_mdot = mcrit(B[i]) # cgs
                    # if we have exceeded the critical value, readjust for magnetic field suppression
                    Mdot_NS = critical_mdot if Mdot_Rmag > critical_mdot else Mdot_Rmag
                    # add the new matter to decay the B field
                    decay_law.decay_field(Mdot_NS * deltaT[i], Rmag) # B[i-1]
                    B[i + 1] =  decay_law.B
                # Rmag at Isco already, there's no B decay## and we assume the mass doesn't make it to the poles, 
                # so Macc going towards the decay does not vary
                else:
                    B[i + 1] = B[i]
                    Mdot_NS = Mdot_Rmag
                    
            T_brake = braking_torque(mu, neutron_star.Rlc)
            T_mag = magnetic_torque(mu, Rmag, fastness, gamma=gamma)
            # update the period so that we get a new Rco and Risco as well as MdotEdd
            neutron_star.torque(T_disc, T_mag, T_brake, deltaT[i])
            
            P_t[i + 1] = neutron_star.P_NS
    except Exception:
        return P_t

    return P_t


conversion = ((1 * (u.g/u.s)).to(u.Msun / u.yr)).value