#!/usr/bin/env python
# coding: utf-8
import numpy as np
from math import log, sqrt
import astropy.units as u
from astropy.constants import G, c, M_sun
from numba import jit, float64

Gcgs = G.to(u.cm**3/u.g/u.s**2).value
ccgs = c.to(u.cm/u.s).value
M_suncgs = M_sun.to(u.g).value

M_NS_default = 2.7837738189772707e+33 # 1.4 M_sun in grams


def accretion_efficiency(M, R):
    """Returns the accretion efficiency. Everything in cgs
    M: float
        Mass of the compact object in g
    R: float
        Radius of the compact object or innermost stable orbit in cm
    Returns the accretion efficiency
    """
    return Gcgs * M  / (2 * ccgs ** 2 * R)


def accretion_luminosity(M_dot, M=1.4 * M_sun, R=10**6 * u.cm):
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

def inverse_magnetospheric_radius(Mdot, R, ns, psi=0.5):

    return ((R/psi)**7 * 2 * Gcgs * ns.M * Mdot**2 / (ns.R_NS**12.)) ** (0.25) # 1/4


@jit(nopython=True)
def mass_transfer_inner_radius(m_0, e_wind=0.5):
    """Equation 23 from Poutanen et al 2007

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor
    e_wind: float
        Fraction of radiative energy that goes to accelerate the outflow
    Returns the mass-transfer rate at the inner radius of the disk in units of m_0 (i.e. fraction of m_0 that makes it to the BH or NS)
    """
    a = e_wind * (0.83 - 0.25 * e_wind)
    # 2/5 = 0.4
    return (1. - a) / (1. - a * (0.4 * m_0) ** (- 0.5))

@jit(nopython=True)
def spherization_radius(m_0, Rin, e_wind=0):
    """
    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor in Eddington units
    Rin: float,
        Inner radius of the disc
    """
    return 5/3 * m_0 * Rin

@jit(nopython=True)
def spherization_radius_poutanen(m_0, Rin, e_wind=0.5):
    """Equation 21 from Poutanen et al 2007.

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor in Eddington units
    Rin: float,
        Inner radius of the disc
    e_wind: float, default 0.5
        Fraction of radiative energy that goes to accelerate the outflow
    Returns Rsph in cm
    """
    return (1.34 - 0.4 * e_wind + 0.1 * e_wind ** 2. - (1.1 - 0.7 * e_wind) * m_0 ** (-2./3.)) * m_0 * Rin

@jit(nopython=True)
def magnetospheric_radius(Mdot, mu, M_NS=M_NS_default, psi=0.5):
    """Return the magnetospheric radius given by my thesis (dipole magnetic field) Equation 2.19. Units cgs (remember Gauss is cgs)

    Parameters:
    -----------
    Mdot: float
        Mass accretion rate in g/s
    mu: float
        Magnetic moment in cgs
    M_NS: float,
        Mass of the NS in g
    psi:float
        Geometrical factor
    Returns the magnetospheric radius (in cm)
    """
    return psi * ( mu**4. / (2. * Gcgs * M_NS * Mdot**2.)) ** (1./7.)


@jit(nopython=True)
def mass_transfer_rate_mag_radius(Mdot, mu, Medd, R_sph, M_NS=M_NS_default, psi=0.5, e_wind=0.5):
    """Solves Equation (6) from Mushtukov et al. 2019 numerically, where R is replaced by Rmag. All parameters in cgs

    Parameters
    ----------
    Mdot: float,
        The mass transfer rate at the companion in Eddington units
    mu: float,
        The magnetic moment in cgs
    Medd: float
        Eddington mass accretion rate in g/s
    R_sph: float,
        Spherization radius
    M_NS: float,
        Mass of the NS in g
    psi: float,
        The numerical factor between 1 and 0 that enters the magnetospheric calculation
    e_wind: float,
        A factor between 0 and 1 that determines the fraction of energy that goes into powering the wind

    Returns the mass-transfer rate at the magnetospheric radius by solving Equation (6) numerically
    """
    mdot = Mdot / Medd
    M_isco = mass_transfer_inner_radius(mdot, e_wind) * Mdot
    M_a = M_isco
    M_b = Mdot 
    err_tol = Mdot * 10.**(-6.)
    err = (M_b - M_a) / 2.
    Rmag_a = magnetospheric_radius(M_a, mu, M_NS, psi=psi)
    M_r = M_isco + (Mdot - M_isco) * Rmag_a / R_sph
    f_a = M_a - M_r
    
        
    while err > err_tol:
        M_c = (M_a + M_b) / 2.
        Rmag_c = magnetospheric_radius(M_c, mu, M_NS, psi=psi)
        M_r = M_isco + (Mdot - M_isco) * Rmag_c / R_sph
        f_c = M_c - M_r
        if f_c * f_a <0:
            M_b = M_c
        else:
            M_a = M_c
            f_a = f_c
        err = abs(M_a - M_b) / 2.

    return M_c

@jit(nopython=True)
def magnetospheric_radius_superEdd(mu, M_NS=M_NS_default, psi=0.5, sph_factor=5./3.):
    """Return the magnetospheric radius when the disk is supercritical. Units cgs (remember Gauss is cgs)
        Mdot(R) = Mdot_0 x R / R_sph
        Rmag = psi * ((Rns**12 * B**4) / (2 * Gcgs * M * (Mdot_0 x R_mag / R_sph)**2)) ** (1/7)
        then bring Rmag**2/7 to the left hand side and solve
        The spherization radius is replaced by ~5/3 x M/Mdot_edd x Rin and Mdot_edd = Ledd / etaxc^2
        Rin goes away as eta also depends on it

        Parameters:
        -----------
        mu: float
            The magnetic moment (typically (B * R_NS**3) / 2) cgs units
        M_NS: float
            Mass of the NS in grams
        psi:float
            Geometrical factor
        sph_factor:
            Spherization factor (R_sph = factor x mdot x Risco, usually use the default)
        Returns the magnetospheric radius (in cm)
    """
    M_NS_sun = M_NS / M_suncgs
    return psi **(7./9.) * (mu**4. / (2. * Gcgs * M_NS)) ** (1./9.) * (sph_factor * Gcgs * M_NS / (2. * eddington_luminosity(M_NS_sun)))**(2./9.)


@jit(nopython=True)
def magnetospheric_radius_superEdd_Mdot(mu, Mdot, Rsph, M_NS=M_NS_default, psi=0.5):
    """Return the magnetospheric radius when the disk is supercritical. Units cgs (remember Gauss is cgs)
    Although it is independent of Mdot0, we preserve the same signature
        Mdot(R) = Mdot_0 x R / R_sph
        Rmag = psi * ((Rns**12 * B**4) / (2 * Gcgs * M * (Mdot_0 x R_mag / R_sph)**2)) ** (1/7)
        then bring Rmag**2/7 to the left hand side and solve
        The spherization radius is replaced by ~5/3 x M/Mdot_edd x Rin and Mdot_edd = Ledd / etaxc^2
        Rin goes away as eta also depends on it

        Parameters:
        -----------
        mu: float
            The magnetic moment (typically (B * R_NS**3) / 2) cgs units
        Mdot: float
            Mass transfer rate in grams
        Rsph: float,
            Spherization radius
        psi:float
            Geometrical factor
        sph_factor:
            Spherization factor (R_sph = factor x mdot x Risco, usually use the default)
        Returns the magnetospheric radius (in cm)
    """
    return psi **(7./9.) * (mu**4 / (2. * Gcgs * M_NS)) ** (1./9.) * (Rsph / Mdot)**(2./9.)


@jit(nopython=True)
def magnetospheric_radius_wang_superEdd(Mdot, mu, Rsph, Rco, M_NS=M_NS_default, gamma=0.1, alpha=0.1):
    """Magnetospheric radius derived using Wang's balance between magnetic and viscous stresses, and accounting for linear mass-loss in the disc.
    The final ecuation is a polynomial of third order: x^3 + K/Rco^(3/2)x - K
    where x = Rmag^(3/2). This can be solved analytically using Cardano's formula
    
    Parameters
    ---------
    Mdot: float,
        Mdot at the donor
    mu: float
            The magnetic moment (typically (B * R_NS**3) / 2) cgs units
    Rsph: float,
        Spherization radius
    Rco: float,
        Co-rotation radius
    M_NS: float
        Mass of the NS in grams
    gamma: float,
        See Wang's 95 for the meaning of this parameter
    alpha: float,
        Alpha for the alpha-prescription (typically <<1)
    """
    Rmag = magnetospheric_radius(Mdot, mu, M_NS, psi=1)
    K = 2 * sqrt(2) / 3 * Rsph * Rmag**(3.5) * gamma/alpha
    Rco_3_2 = (Rco)**(1.5)
    q = -K
    p = K/Rco_3_2
    determinant = np.sqrt(q**2 / 4 + p**3. / 27)
    # python sometimes gives the complex result of **1/3 for big numbers, let's take the real value by taking the abs
    R0 = ( (-q/2 + determinant)**(1/3) + abs(-q/2 - determinant)**(1/3))  **(2/3)
    return R0


@jit(nopython=True)
def magnetospheric_radius_wang(Mdot0, mu, M_NS, Rco, gamma=0.1, alpha=0.1):
    """Magnetospheric radius derived using Wang's balance between magnetic and viscous stresses.
    The final ecuation can only be solved numerically. This can be solved analytically using Cardano's formula

    Parameters
    ---------
    
    Mdot0: float,
        Mdot at the donor
    mu: float
            The magnetic moment (typically (B * R_NS**3) / 2) cgs units
    M_NS: float
        Mass of the NS in grams
    Rco: float,
        The co-rotation radius
    gamma: float,
        See Wang's 95 for the meaning of this parameter
    alpha: float,
        Alpha for the alpha-prescription (typically <<1)
    """
    def f_R0(R0, K, Rco_3_2):
        return R0**(3.5) + K/Rco_3_2 * R0**(1.5) - K

    Rmag = magnetospheric_radius(Mdot0, mu, M_NS, psi=1)
    K = 2 * Rmag**(3.5) * gamma/alpha
    Rco_3_2 = (Rco)**(1.5)
    
    R0_a = 10**6. # ns radius
    R0_b = Rco * 10
    err_tol = 1000 # 1 km
    err = (R0_b - R0_a) / 2
    f_a = f_R0(R0_a, K, Rco_3_2)
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
    
@jit(nopython=True)
def mcrit(B):
    """Calculate mdot critical at which the magnetic confiment breaks due to radiation pressure according to Mushtukov. cgs units everywhere
    float or array: B
        Magnetic field strength (assumed in G)
    Returns the critical Mdot (in Mcrt (g/s))
    """
    return np.exp(6.9233445 + 4.2990807 * np.log(B) - 0.1794699 * np.log(B)**2. + 0.0025782 * np.log(B)**3.)


@jit(nopython=True)
def eddington_luminosity(Msuns):
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        Msuns: float
            Mass in solar units
        Returns the Eddington luminosity in erg/s (cgs)
    """
    return 1.26 * Msuns * 10**38.


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
    return eddington_luminosity(M).decompose(bases=[u.g, u.s, u.cm]) / efficiency / c.to(u.cm/u.s)**2.



@jit(nopython=True)
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
    factor =  170 * (viscosity_alpha / 0.1) ** (2/9) * m_ns ** (-10/9.)
    magnetic_moment = (Rin / factor) ** (9/4)
    return magnetic_moment_to_B(magnetic_moment * 10**30 * u.G * u.cm ** 3)


def magnetic_moment_to_B(mu, Rns = 10 * u.km):
    """Compute magnetic field from magnetic moment. See after Equation (2) from Tsygankov et al 2016
        Returns the magnetic field in G units."""
    B = 2 * mu.to(u.G * u.km ** 3) / (Rns.to(u.km)) ** 3
    return B

@jit(nopython=True)
def fastness_parameter(Rmag, Rco):
    """Computes the fastness parameter. Both radii must be provided in same units

    Parameters
    ----------
    Rm: float or array-like
        Magnetospheric radius
    Rco: float or array-like
        Co-rotation radius
    """
    return (Rmag / Rco) ** (1.5)

@jit(nopython=True)
def torque_wang(Mdot, Rmag, Rco, M_NS=M_NS_default):
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
    n = (7/6 - (4/3) * omega + (1/9)*omega**2.) / (1 - omega)
    return N_0 * n

@jit(nopython=True)
def magnetic_torque_wang(Mdot, Rmag, Rco, M_NS=M_NS_default):
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
    return N_0 / (1 - omega) * (1/6 - omega/3 + omega**2./9)

@jit(nopython=True)
def accretion_torque(Mdot, Rmag, M_NS=M_NS_default):
    """Computes the accretion torque onto a NS

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g. Defaults to 1.4 M_sun in grams
    """
    N_acc = Mdot * np.sqrt(Gcgs * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc

@jit(nopython=True)
def accretion_torque_dai(Mdot, Rmag, omega, M_NS=M_NS_default, gamma=1, delta=0.1, psi=0.5):
    """Computes the accretion torque onto a NS accounting for the magnetosphere interaction (Eq. 10 from Lai & Li 2006)

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    omega: float
        Fastness parameter
    M_NS: float
        Mass of the NS in g
    float: xi,
        Factor to account for not full transfer of angular momentum
    """
    xi = sqrt(2.) * gamma * delta
    N_acc = xi * accretion_torque(Mdot, Rmag, M_NS) * (1. - omega) / (psi**(3.5)) # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc

@jit(nopython=True)
def magnetic_torque_dai(mu, Rmag, omega, gamma=1):
    """Computes the magnetic torque onto a NS (valid during accretion) i.e. for w >=1 (Lai & Li 2006)
        Their Equation 7

    Parameters
    ----------
    mu: float
        Magnetic moment
    Rmag: float
        Magnetospheric radius in cm
    omega: float
        Fastness parameter
    gamma: float,
        Factor to account from Equation 4
    """
    factor = gamma * mu**2. / (3. * Rmag**3)
    Nmag = factor * (1. - 2. * omega + 2. * omega**2./3.)
    return Nmag

@jit(nopython=True)
def magnetic_torque_dai_propeller(mu, Rmag, omega, gamma=1):
    """Computes the magnetic torque onto a NS (valid during propeller i.e. for w <1) (Lai & Li 2006)
    Their Equation 7

    Parameters
    ----------
    mu: float
        Magnetic moment
    Rmag: float
        Magnetospheric radius in cm
    omega: float
        Fastness parameter
    gamma: float,
        Factor to account from Equation 4
    """
    factor = gamma * mu**2. / (3. * Rmag**3.)
    Nmag = factor * (2. /(3. * omega) - 1.)
    return Nmag

@jit(nopython=True)
def propeller_torque(Mdot, M_NS, omega, Rm):
    """Equation 12 from Illarionov & Sunyaev 1975 or Eq 42 from Abolmasov 2024 review
    
    Parameters
    ----------
    omega: float,
        Angular velocity of the NS (2 pi / P)
    Rm :float,
        Magnetospheric radius
    """
    return -Mdot * Gcgs * M_NS / Rm / omega

@jit(nopython=True)
def magnetic_moment(B, R_NS=10**6):
    """See e.g. Equation (2) from Tsygankov.
    Everything in cgs (Gauss and cm)

    Parameters
    ----------
    B: float,
        Magnetic field in Gauss
    R_NS:float,
        Radius of the NS
    """
    return B * R_NS**3. / 2.