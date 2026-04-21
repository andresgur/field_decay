#!/usr/bin/env python
# coding: utf-8
from numpy import log, exp
from .constants import Gcgs, ccgs
from math import pi
from numba import jit, njit, float64
from scipy.optimize import brentq

# jit does make the code faster


M_NS_default = 2.7837738189772707e33  # 1.4 M_sun in grams


def inverse_magnetospheric_radius(Mdot, R, ns, psi=0.5):
    """Return the inverse magnetospheric radius given by my thesis (dipole magnetic field) Equation 2.19. Units cgs (remember Gauss is cgs)

    Parameters:
    -----------
    Mdot: float
        Mass accretion rate in g/s
    R: float
        Radius in cm
    ns: object
        Neutron star object with attributes M (mass) and R_NS (radius)
    psi:float
        Geometrical factor
    Returns the inverse magnetospheric radius (in cm)
    """
    return ((R / psi) ** 7 * 2.0 * Gcgs * ns.M * Mdot**2 / (ns.R_NS**12.0)) ** (
        0.25
    )  # 1/4


@jit(nopython=True)
def neutron_star_binding_luminosity(
    Mdotmag, Rmag, M_NS=M_NS_default, R_NS=10**6, beaming=1
):
    """Computes the luminosity within Rmag (See Middleton+2022 Equation 2)

    Mdotmag: astropy.quantity
        Mass-accretion rate at the magnetospheric radius
    """
    return (Gcgs * M_NS * Mdotmag) * (1.0 / R_NS - 1.0 / Rmag) / beaming


@jit(nopython=True)
def luminosity_super_edd_NS(Rmag, Rsph, e_wind=0.5, beaming=1):
    """Equations from Middleton+23. This is the luminosity from the supercritical disc and therefore assumes Rsph > Rmag. All parameters are in cgs

    Parameters
    ----------
    Rmag:float,
        Magnetosphericc radius
    Rsph: float,
        Spherization radius,
    e_wind: float
        Energy imparted to the wind (0 < e_wind < 1)
    beaming: float
        Beaming factor (b < 1, b=1 no beaming)

    Returns the luminosity in Eddington units (i.e. L / LEdd)
    """
    return log(Rsph / Rmag) * (1.0 - e_wind) / beaming


@jit(nopython=True)
def mass_transfer_inner_radius(m_0: float, e_wind: float = 0.5) -> float:
    """Equation 23 from Poutanen et al 2007. Only valid for m_0 > 2.5, if below returns 1

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor. Must be below 2.5 otherwise return the same value (i.e. assume all mass makes it to the CO)
    e_wind: float
        Fraction of radiative energy that goes to accelerate the outflow
    Returns
    -------
    float: The mass-transfer rate at the inner radius of the disk in units of m_0 (i.e. fraction of m_0 that makes it to the BH or NS)
    Note: this returns 1 if mdot is below 2.5 (as the approximation does not hold there) and we assume all mass makes it to the CO

    """

    if m_0 < 2.5:
        return 1
    a = e_wind * (0.83 - 0.25 * e_wind)
    # 2/5 = 0.4
    return (1.0 - a) / (1.0 - a * (0.4 * m_0) ** (-0.5))


@jit(nopython=True)
def spherization_radius(m_0, e_wind=0):
    """
    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor in Eddington units
    Rin: float,
        Inner radius of the disc
    Returns
    ------
    The spherization radius in inner radius units
    """
    return 5.0 / 3.0 * m_0


@jit(nopython=True)
def spherization_radius_poutanen(m_0: float, e_wind: float = 0.5) -> float:
    """Equation 21 from Poutanen et al 2007.

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor in Eddington units
    Rin: float,
        Inner radius of the disc
    e_wind: float, default 0.5
        Fraction of radiative energy that goes to accelerate the outflow
    Returns in units of inner radius
    """
    return (
        1.34
        - 0.4 * e_wind
        + 0.1 * e_wind**2.0
        - (1.1 - 0.7 * e_wind) * m_0 ** (-2.0 / 3.0)
    ) * m_0


@jit(nopython=True)
def magnetospheric_radius(
    Mdot: float, mu: float, M_NS: float = M_NS_default, psi: float = 0.5
) -> float:
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
    return psi * (mu**4.0 / (2.0 * Gcgs * M_NS * Mdot**2.0)) ** (1.0 / 7.0)


@njit
def rmag_inclination_factor(chi=0, integration_points=10000):
    """Inclination factor for the magnetospheric radius of an inclined (magnetic axis wrt to disk) accretor (Jetzer 1998; Eq 6).
    Instead of dealing of an azimuthally asymetric Rmag, we calculate the average factor to correct it
    Parameters
    ----------
    chi: float,
        Angle (radians) between the magnetic and disc axis (Default to 0, i.e. aligned rotator).
    integration_points: int,
        Number of points to use for the numerical integration between 0 and 2pi. Default 10000
    Returns the inclination factor to be applied to the magnetospheric radius
    """
    import numpy as np

    x = np.linspace(0, 2 * pi, integration_points)
    y = (1 + 3 * (np.sin(chi) * np.sin(x)) ** 2) ** (2 / 7)
    average = np.trapz(y, x) / (2 * pi)
    return average


@njit
def _Mdot_root(Mdot, Mdot0, Mdotisco, R_sph, mu, M_NS=M_NS_default, psi=0.5):
    """Auxiliary for the numerical solver of the mass transfer rate at the magnetospheric radius.
    This is Equation (6) from Mushtukov et al. 2019, but with R replaced by Rmag

    Parameters
    ----------
    Mdot: float,
        The mass transfer rate at the magnetospheric radius in g/s
    Mdot0: float,
        The mass transfer rate at the companion
    Mdotisco: float,
        The mass transfer rate at the inner radius of the disk
    mu: float,
        The magnetic moment in cgs
    R_sph: float,
        The spherization radius in cm
    M_NS: float,
        The mass of the NS in g
    psi: float,
        The numerical factor between 1 and 0 that enters the magnetospheric calculation
    """
    Rmag_a = magnetospheric_radius(Mdot, mu, M_NS, psi=psi)
    M_r = Mdotisco + (Mdot0 - Mdotisco) * Rmag_a / R_sph
    f = Mdot - M_r
    return f / Mdot0


# we cannot use decorators here because of kwargs
@njit
def secant_method(func, x0, x1, tol=1e-4, max_iter: int = 100, *args):
    """
    Finds the root of a function using the Secant method. x0<x1

    The Secant method is a root-finding algorithm that uses a succession of roots
    of secant lines to better approximate a root of a function $f$.


    The next approximation $x_2$ is calculated using the formula:
    $$
    x_2 = x_1 - f(x_1) \frac{x_1 - x_0}{f(x_1) - f(x_0)}
    $$

    Parameters
    ----------
    func : callable
        The function $f(x)$ for which to find the root (i.e. finding x for which f(x) = 0). It must
        accept at least one numerical argument $x$ and any keyword arguments
        specified in **kwargs.
    x0 : float
        The first initial guess for the root.
    x1 : float
        The second initial guess for the root.
    tol : float, optional
        The tolerance for convergence. Iteration stops when
        $|x_2 - x_1| < \text{tol}$. Defaults to 1e-4.
    max_iter : int, optional
        The maximum number of iterations allowed.
        Defaults to 100.
    **kwargs : dict
        Additional keyword arguments to be passed to the function `func`
        at each evaluation
    Returns
    -------
    float
        The approximate root of the function $f(x)$ i.e. x for which $f(x)=0$. None if it did not find a root before the iterations
    """
    for i in range(max_iter):
        f_0 = func(x0, *args)
        f_1 = func(x1, *args)
        x2 = x1 - f_1 * (x1 - x0) / (f_1 - f_0)
        if abs(x2 - x1) < tol:
            return x2
        # 0 --> 1
        # 1 -->2
        x0, x1 = x1, x2

    return None


@jit(nopython=True)
def mass_transfer_rate_mag_radius_brent(
    Mdot0,
    mu,
    Mdot_isco,
    R_sph,
    M_NS=M_NS_default,
    psi=0.5,
    tol=1e-4,
    max_iter: int = 500,
):
    """Solves Equation (6) from Mushtukov et al. 2019 numerically, where R is replaced by Rmag. All parameters in cgs

    Parameters
    ----------
    Mdot0: float,
        The mass transfer rate at the companion in Eddington units
    mu: float,
        The magnetic moment in cgs
    Mdot_isco: float,
        The mass transfer rate at the inner stable circular orbit in g/s
    R_sph: float,
        Spherization radius
    M_NS: float,
        Mass of the NS in g
    psi: float,
        The numerical factor between 1 and 0 that enters the magnetospheric calculation
    tol: float,
        Tolerance for the numerical solution in units of Mdot0. Default 10^-4 (i.e. 10^-4 Mdot0). Some tests were run to ensure that this is enough to get a good convergence

    Returns the mass-transfer rate at the magnetospheric radius by solving Equation (6) numerically
    """
    Rmag = magnetospheric_radius(Mdot0, mu, M_NS, psi)
    if Rmag > R_sph:
        return Mdot0
    res = brentq(
        _Mdot_root,
        Mdot_isco,
        Mdot0,
        args=(Mdot0, Mdot_isco, R_sph, mu, M_NS, psi),
        xtol=tol,
        maxiter=max_iter,
    )
    return res


@njit
def mass_transfer_rate_mag_radius_secant(
    Mdot0,
    mu,
    Mdot_isco,
    R_sph,
    M_NS=M_NS_default,
    psi=0.5,
    tol=1e-1,
    max_iter: int = 500,
) -> float:
    """Solves Equation (6) from Mushtukov et al. 2019 numerically, where R is replaced by Rmag. All parameters in cgs

    Parameters
    ----------
    Mdot0: float,
        The mass transfer rate at the companion in Eddington units
    mu: float,
        The magnetic moment in cgs
    Mdot_isco: float,
        The mass transfer rate at the inner stable circular orbit in g/s
    R_sph: float,
        Spherization radius
    M_NS: float,
        Mass of the NS in g
    psi: float,
        The numerical factor between 1 and 0 that enters the magnetospheric calculation
    tol: float,
        Tolerance for the numerical solution in units of Mdot0. Default 10^-1. Some tests were run to ensure that this is enough to get a good convergence (it is fairly encompassing as Mdot is of the order of 10**18 g/s)

    Returns the mass-transfer rate at the magnetospheric radius by solving Equation (6) numerically
    """
    Rmag = magnetospheric_radius(Mdot0, mu, M_NS, psi)
    if Rmag > R_sph:
        return Mdot0
    # this returns 1 if mdot is below 2.5 (as the approximation does not hold there) and we assume all mass makes it to the CO
    if Mdot_isco / Mdot0 == 1.0:
        return Mdot0

    M_1 = Mdot_isco
    M_0 = Mdot0
    M_2 = secant_method(
        _Mdot_root, M_0, M_1, tol, max_iter, Mdot0, Mdot_isco, R_sph, mu, M_NS, psi
    )

    return M_2


@njit
def mass_transfer_rate_mag_radius_bisection(
    Mdot0,
    mu,
    M_isco,
    R_sph,
    M_NS=M_NS_default,
    psi=0.5,
    err_tol=1e-2,
    max_iter: int = 200,
):
    """Solves Equation (6) from Mushtukov et al. 2019 numerically, where R is replaced by Rmag. All parameters in cgs

    Parameters
    ----------
    Mdot0: float,
        The mass transfer rate at the companion in Eddington units
    mu: float,
        The magnetic moment in cgs
    M_isco: float
        The mass transfer rate at the inner stable circular orbit in g/s
    R_sph: float,
        Spherization radius
    M_NS: float,
        Mass of the NS in g
    psi: float,
        The numerical factor between 1 and 0 that enters the magnetospheric calculation
    err_tol: float,
        Tolerance for the numerical solution in units of Mdot0. Default 10^-2. Some tests were run to ensure that this is enough to get a good convergence
    max_iter: int,
        Maximum number of iterations for the bisection method. Default 200.

    Returns the mass-transfer rate at the magnetospheric radius by solving Equation (6) numerically
    """

    M_b = M_isco
    Rmag = magnetospheric_radius(Mdot0, mu, M_NS, psi)
    if Rmag > R_sph:
        return Mdot0
    M_a = Mdot0
    f_a = _Mdot_root(M_a, Mdot0, M_isco, R_sph, mu, M_NS, psi)
    # calculate the M_c here in case Mdot = Mdotisco (this happens if mdot is below 2.5, as the approximation does not hold there. In this case we simply return the same Mdot0)
    for _ in range(max_iter):
        M_c = (M_a + M_b) / 2.0

        f_c = _Mdot_root(M_c, Mdot0, M_isco, R_sph, mu, M_NS, psi)
        if f_c * f_a < 0:
            M_b = M_c
        else:
            M_a = M_c
            f_a = f_c
        # b is always larger than a
        err = (M_a - M_b) / 2.0
        if err < err_tol or f_c == 0:
            return M_c

    return None


@jit(nopython=True)
def magnetospheric_radius_superEdd(neutron_star, psi=0.5, sph_factor=5.0 / 3.0):
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
    M_NS = neutron_star.M
    Ledd = neutron_star.Ledd
    mu = neutron_star.mu
    return (
        psi ** (7.0 / 9.0)
        * (mu**4.0 / (2.0 * Gcgs * M_NS)) ** (1.0 / 9.0)
        * (sph_factor * Gcgs * M_NS / (2.0 * Ledd)) ** (2.0 / 9.0)
    )


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
    return (
        psi ** (7.0 / 9.0)
        * (mu**4 / (2.0 * Gcgs * M_NS)) ** (1.0 / 9.0)
        * (Rsph / Mdot) ** (2.0 / 9.0)
    )


@jit(nopython=True)
def magnetospheric_radius_wang_superEdd(
    Mdot, mu, Rsph, Rco, M_NS=M_NS_default, gamma=0.1, alpha=0.1
):
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
    K = 2.0 * 2.0**0.5 / 3.0 * Rsph * Rmag ** (3.5) * gamma / alpha  # sqrt(2)
    Rco_3_2 = (Rco) ** (1.5)
    q = -K
    p = K / Rco_3_2
    determinant = (q**2.0 / 4.0 + p**3.0 / 27.0) ** 0.5
    # python sometimes gives the complex result of **1/3 for big numbers, let's take the real value by taking the abs
    R0 = (
        (-q / 2 + determinant) ** (1.0 / 3.0)
        + abs(-q / 2.0 - determinant) ** (1.0 / 3.0)
    ) ** (2.0 / 3.0)
    return R0


@jit(nopython=True)
def magnetospheric_radius_wang(Mdot0, mu, M_NS, Rco, gamma=0.1, alpha=0.1, psi=1):
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
        return R0 ** (3.5) + K / Rco_3_2 * R0 ** (1.5) - K

    Rmag = magnetospheric_radius(Mdot0, mu, M_NS, psi=psi)
    K = 2 * Rmag ** (3.5) * gamma / alpha
    Rco_3_2 = (Rco) ** (1.5)

    R0_a = 10**6.0  # ns radius
    R0_b = Rco * 10.0
    err_tol = 100.0  # 1 km
    err = (R0_b - R0_a) / 2
    f_a = f_R0(R0_a, K, Rco_3_2)
    while err > err_tol:
        R0_c = (R0_a + R0_b) / 2
        f_c = f_R0(R0_c, K, Rco_3_2)

        if f_c * f_a < 0:
            R0_b = R0_c
        else:
            R0_a = R0_c
            f_a = f_c
        err = abs(R0_a - R0_b) / 2
    return R0_c, err


@jit(float64(float64), nopython=True)
def mcrit(B):
    """Calculate mdot critical at which the magnetic confiment breaks due to radiation pressure according to Mushtukov. cgs units everywhere
    float or array: B
        Magnetic field strength (assumed in G)
    Returns the critical Mdot (in Mcrt (g/s))
    """
    logB = log(B)
    return exp(
        6.9233445 + 4.2990807 * logB - 0.1794699 * logB**2.0 + 0.0025782 * logB**3.0
    )


@jit(nopython=True)
def magnetic_moment_to_B(mu, Rns=1e6):
    """Compute magnetic field from magnetic moment. See after Equation (2) from Tsygankov et al 2016
    Returns the magnetic field in G units.
    Parameters
    ----------
    mu: float
        Magnetic moment in cgs units (i.e. Gauss * cm^3)

    Rns: float
        Radius of the neutron star in cm. Default 10^6 cm (1 km)"""
    B = 2.0 * mu / (Rns) ** 3.0
    return B


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
    factor = (
        170.0 * (viscosity_alpha / 0.1) ** (2.25) * m_ns ** (-10 / 9.0)
    )  # 9/4 = 2.25
    mu = (Rin / factor) ** (2.25)
    B = magnetic_moment_to_B(mu * 1e30)  #  * u.G * u.cm ** 3
    return B


@jit(
    float64(float64, float64),
    nopython=True,
)
def fastness_parameter(Rmag: float, Rco: float) -> float:
    """Computes the fastness parameter. Both radii must be provided in same units

    Parameters
    ----------
    Rm: float or array-like
        Magnetospheric radius
    Rco: float or array-like
        Co-rotation radius
    """
    return (Rmag / Rco) ** (1.5)


@njit
def magnetic_moment(B: float, R_NS: float = 10**6):
    """See e.g. Equation (2) from Tsygankov.
    Everything in cgs (Gauss and cm)

    $mu = B * R_NS**3 / 2$
    """
    return B * R_NS**3.0 / 2.0


def scale_height(m_r, R, R0):
    """Equation 18 from Lipunova+99 (simplified for the acc efficiency), works for both sub and super critical disks as long as advection is neglected
        Just replace Mdot(R) by the appropiate calculation (i.e. without or with outflows)
        Everything in cgs units.
        H = Rg * m_r * 3 / 4 / efficiency * (1 - (R0 / R) ** 0.5)
        efficiency = Rg / (2R0)
        H = Rg * m_r * 3 / 4 / (Rg / (2R0)) * (1 - (R0 / R) ** 0.5)
        H = m_r * 3 / 2 * R0 * (1 - (R0 / R) ** 0.5)

    Parameters
    ----------
    m_r:float
        (Dimensionless) Mass-transfer rate at every radii (or at a given radius R) in Eddington units
    R: float or array
        Radius or radii at which the scale height is to be calculated
    R0: float
        Inner radius of the disk (typically isco)


    """
    H = m_r * 3 / 2 * R0 * (1 - (R0 / R) ** 0.5)
    return H
