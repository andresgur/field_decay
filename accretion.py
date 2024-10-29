#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math as m
import astropy.units as u
from astropy.constants import G, c, M_sun


Gcgs = G.to(u.cm**3/u.g/u.s**2).value

ccgs = c.to(u.cm/u.s).value


def accretion_efficiency(M, R):
    """Returns the accretion efficiency. Everything in cgs
    M: float
        Mass of the compact object in g
    R: float
        Radius of the compact object or innermost stable orbit in cm
    Returns the accretion efficiency
    """
    return Gcgs * M  / (2 * ccgs ** 2 * R)


def accretion_luminosity(M_dot, M=1.4 * M_sun.to(u.g), R=10**6 * u.cm):
    """Returns the accretion luminosity in erg/s (see Vasilopoulos et al 2019 paragraph after eq 8.
        M_dot: astropy.quantity,
            Mass-accretion rate in g/s
        M: astropy.quantity,
            Mass of the compact object in solar masses
        R: astropy.quantity,
            Radius of the compact object or inner stable orbit
        Returns the accretion luminosity in erg/s
    """
    efficiency = accretion_efficiency(M.to(u.g).value, R.to(u.cm).value)
    return efficiency * M_dot * c.to(u.cm/u.s) ** 2


def magnetospheric_radius(Mdot, B, NS, psi=0.5):
    """Return the magnetospheric radius given by my thesis (dipole magnetic field) Equation 2.19. Units cgs (remember Gauss is cgs)

        Parameters:
        -----------
        Mdot: float
            Mass accretion rate in g/s
        B: float
            Magnetic field in Gauss
        M: float,
            Mass of the NS in g
        Rns: float,
            Radius of the NS in cm
        psi:float
            Geometrical factor
        Returns the magnetospheric radius (in cm)
    """
    return psi * ((NS.R_NS**12 * B**4) / (2 * Gcgs * NS.M * Mdot**2)) ** (1/7)


def magnetospheric_radius_wang_superEdd(Mdot0, B, NS, gamma=0.1, alpha=0.1):
    """Magnetospheric radius derived using Wang's balance between magnetic and viscous stresses, and accounting for linear mass-loss in the disc.
    The final ecuation is a polynomial of third order: x^3 + K/Rco^(3/2)x - K
    where x = Rmag^(3/2). This can be solved analytically using Cardano's formula
    
    Parameters
    ---------
    Mdot0: float,
        Mdot at the donor
    NS: compact_object.NS
        NS object
    gamma: float,
        See Wang's 95 for the meaning of this parameter
    alpha: float,
        Alpha for the alpha-prescription (typically <<1)
    """
    Rmag = magnetospheric_radius(Mdot0, B, NS, psi=1)
    Rsph = 5 / 3 * Mdot0 / NS.Medd * NS.Risco
    K = 2 * m.sqrt(2) / 3 * Rsph * Rmag**(7/2) * gamma/alpha
    Rco_3_2 = (NS.Rco)**(3/2)
    q = -K
    p = K/Rco_3_2
    determinant = np.sqrt(q**2 / 4 + p**3 / 27)
    # python sometimes gives the complex result of **1/3 for big numbers, let's take the real value by taking the abs
    R0 = ( (-q/2 + determinant)**(1/3) + abs(-q/2 - determinant)**(1/3))  **(2/3)
    return R0

def magnetospheric_radius_wang(Mdot0, B, NS, gamma=0.1, alpha=0.1):
    """Magnetospheric radius derived using Wang's balance between magnetic and viscous stresses.
    The final ecuation can only be solved numerically. This can be solved analytically using Cardano's formula

    Parameters
    ---------
    
    Mdot0: float,
        Mdot at the donor
    NS: compact_object.NS
        NS object
    gamma: float,
        See Wang's 95 for the meaning of this parameter
    alpha: float,
        Alpha for the alpha-prescription (typically <<1)
    """
    def f_R0(R0, K, Rco_3_2):
        return R0**(7/2) + K/Rco_3_2 * R0**(3/2) - K

    Rmag = magnetospheric_radius(Mdot0, B, NS, psi=1)
    K = 2 * Rmag**(7/2) * gamma/alpha
    Rco_3_2 = (NS.Rco)**(3/2)
    
    R0_a = NS.R_NS
    R0_b = NS.Rco * 10
    err_tol = NS.R_NS * 10 **-5
    err = (R0_b - R0_a) / 2
    f_a = f_R0(R0_a, K, Rco_3_2)
    R0_c = (R0_a + R0_b) / 2
    while err > err_tol:
        R0_c = (R0_a + R0_b) / 2
        f_c = f_R0(R0_c, K, Rco_3_2)
        
        if f_c * f_a <0:
            R0_b = R0_c
        else:
            R0_a = R0_c
            f_a = f_c
        err = abs(R0_a - R0_b) / 2
    return R0_c, err
    


def magnetospheric_radius_superEdd(B, NS, psi=0.5, sph_factor=5/3):
    """Return the magnetospheric radius when the disk is supercritical. Units cgs (remember Gauss is cgs)
        Mdot(R) = Mdot_0 x R / R_sph
        Rmag = psi * ((Rns**12 * B**4) / (2 * Gcgs * M * (Mdot_0 x R_mag / R_sph)**2)) ** (1/7)
        then bring Rmag**2/7 to the left hand side and solve

        Parameters:
        -----------
        NS: compact_object.NS
            A NS object
        B: float
            Magnetic field in Gauss
        psi:float
            Geometrical factor
        sph_factor:
            Spherization factor (R_sph = factor x mdot x Risco, usually use the default)
        Returns the magnetospheric radius (in cm)
    """
    return psi **(7/9) * ((NS.R_NS**12 * B**4) / (2 * Gcgs * NS.M)) ** (1/9) * (sph_factor * NS.Risco/ NS.Medd)**(2/9)


def isco_radius(M, a=0.998):
    """Returns the ISCO radius for a given mass in cm.
        Parameters
        ----------
        M: float,
            Mass of the compact object in cgs
        a: float,
        dimensionless spin: 0 for a Scharzschild black hole or 0.998 for a Kerr black hole.
    Returns the radius of the inner most stable orbit in cm
    """
    z1 = 1 + (1 - a**2) ** (1/3) * ((1 + a)** (1/3) + (1-a) ** (1/3))
    z2 = np.sqrt(3 * a ** 2 + z1**2)
    return (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * gravitational_radius(M)


def schwarzschild_radius(M):
    """Returns the scharzchild radius for a given mass in cm. 
        Parameters
        ----------
        M: float
            In grams
    """
    return 2 * gravitational_radius(M)


def gravitational_radius(M):
    """Returns the gravitational radius for a given mass in g.
    Parameters
    ----------
    M: float
        Mass of the compact object in grams

    Returns the gravitational radius in cm
    """
    return (Gcgs * M/ ccgs**2)


def eddington_luminosity(M):
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: float
            Mass in solar units
        Returns the Eddington luminosity in erg/s (cgs)
    """
    return 1.26 * M * 10**38


def eddington_accretion_rate(M, R_in):
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: astropy.quantity
        R_in: astropy.quantity
        Returns the Eddington accretion rate in quantity
    """
    efficiency = accretion_efficiency(M, R_in)
    # convert erg to cgs
    return eddington_luminosity(M).decompose(bases=[u.g, u.s, u.cm]) / efficiency / c.to(u.cm/u.s)**2


def chashkina_inner_radius(Rin, viscosity_alpha=0.5, m_ns=1):
    """Computes the magnetic field given the inner disk radius in the radiation-pressure dominated state. Equation 29.

        Parameters:
        ----------
        Rin:float
            Inner radius of the disk in R_g
        viscosity_alpha: float
            The alpha parameter from Shakura & Sunyaev disk prescription. Default 0.5 (see Table 1 Chashkina)
        m_ns: float
            NS mass (in NS solar mass units e.g. 1 for 1.4 M_sun)
        Returns the magnetic field in 10¹² G units.
        """
    factor =  170 * (viscosity_alpha / 0.1) ** (2/9) * m_ns ** (-10/9)
    magnetic_moment = (Rin / factor) ** (9/4)
    return magnetic_moment_to_B(magnetic_moment * 10**30 * u.G * u.cm ** 3)


def magnetic_moment_to_B(mu, Rns = 10 * u.km):
    """Compute magnetic field from magnetic moment. See after Equation (2) from Tsygankov et al 2016
        Returns the magnetic field in 10¹² G units."""
    B = 2 * mu.to(u.G * u.km ** 3) / (Rns.to(u.km)) ** 3
    return B / 10**12


def fastness_parameter(Rmag, Rco):
    """Computes the fastness parameter. Both radii must be provided in same units

    Parameters
    ----------
    Rm: float
        Magnetospheric radius
    Rco: float
        Co-rotation radius
    """
    return (Rmag / Rco) ** (3/2)


def torque_wang(Mdot, Rmag, Rco, M_NS=(1.4 * M_sun).to(u.g).value):
    """Computes the accretion torque onto a NS according to Wang+95 (actually taken from Vasilopoulos+2018). All units in cgs

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    Rco: float
        Co-rotation radius in cm
    M_NS: float
        Mass of the NS, defaults to 1.4 solar massess (in g)
    """
    omega = fastness_parameter(Rmag, Rco)
    N_0 = Mdot * np.sqrt(Gcgs * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    n = (7/6 - (4/3) * omega + (1/9)*omega**2) / (1 - omega)
    return N_0 * n


def magnetic_torque_wang(Mdot, Rmag, Rco, M_NS=(1.4 * M_sun).to(u.g).value):
    """This is like the above, but only considering the magnetic term. It is a spin down term due to magnetic field lines threading the disc beyond the co-rotation radius

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    Rco: float
        Co-rotation radius in cm
    M_NS: float
        Mass of the NS, defaults to 1.4 solar massess (in g)
    """
    omega = fastness_parameter(Rmag, Rco)
    N_0 = Mdot * np.sqrt(Gcgs * M_NS * Rmag)
    return N_0 / (1 - omega) * (1/6 - omega/3 + omega**2/9)


def accretion_torque(Mdot, Rmag, M_NS=(1.4 * M_sun).to(u.g).value):
    """Computes the accretion torque onto a NS

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g
    """
    N_acc = Mdot * np.sqrt(Gcgs * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc


def accretion_torque_dai(Mdot, Rmag, NS, gamma=1, delta=1 / m.sqrt(2)):
    """Computes the accretion torque onto a NS accounting for the magnetosphere interaction (Eq. 10 from Lai & Li 2006)

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g
    float: xi,
        Factor to account for not full transfer of angular momentum
    """
    omega = fastness_parameter(Rmag, NS.Rco)
    xi = m.sqrt(2) * gamma * delta
    N_acc = xi * accretion_torque(Mdot, Rmag, NS.M) * (1 - omega) # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc


def magnetic_torque_dai(B, Rmag, NS, gamma=1):
    """Computes the magnetic torque onto a NS (valid during accretion) i.e. for w >=1 (Lai & Li 2006)
        Their Equation 7

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g
    gamma: float,
        Factor to account from Equation 4
    """
    mu = B * NS.R_NS**3
    omega = fastness_parameter(Rmag, NS.Rco)
    factor = gamma * mu**2 / (3 * Rmag**3)
    Nmag = factor * (1 - 2 * omega + 2 * omega**2/3)
    return Nmag


def magnetic_torque_dai_propeller(B, Rmag, NS, gamma=1):
    """Computes the magnetic torque onto a NS (valid during propeller i.e. for w <1) (Lai & Li 2006)
    Their Equation 7

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g
    gamma: float,
        Factor to account from Equation 4
    """
    mu = B * NS.R_NS**3
    omega = fastness_parameter(Rmag, NS.Rco)
    factor = gamma * mu**2 / (3 * Rmag**3)
    Nmag = factor * (2/(3 * omega) - 1)
    return Nmag


def propeller_torque(Mdot, M_NS, omega, Rm):
    """Equation 12 from Illarionov & Sunyaev 1975 or Eq 42 from Abolmasov 2024 review
    
    omega: float,
        Angular velocity of the NS (2 pi / P)
    Rm :float,
        Magnetospheric radius
    """
    return -Mdot * Gcgs * M_NS / Rm / omega



def magnetic_moment(B, R_NS=10*u.km):
    """See e.g. Equation (2) from Tsygankov

    """
    return (B.to(u.G) * R_NS.to(u.cm)**3) / 2