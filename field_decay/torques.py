from math import cos, pi, sin
from numba import jit, njit, float64
from accretion import M_NS_default
from constants import Gcgs, ccgs


@njit
def accretion_torque(Mdot: float, Rmag: float, M_NS: float = M_NS_default) -> float:
    r"""Computes the accretion torque onto a NS.

    $$N = \dot{M} * \sqrt{G \, M \, R{_\mathrm{mag}}}$$

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    M_NS: astropy.quantity
        Mass of the NS, in g. Defaults to 1.4 M_sun in grams
    """
    N_acc = Mdot * (Gcgs * M_NS * Rmag) ** 0.5  # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc


@njit
def torque_wang(
    Mdot: float, Rmag: float, omega: float, M_NS=M_NS_default, chi: float = 0
):
    """Computes the accretion torque onto a NS according to Wang+95 (actually taken from Vasilopoulos+2018).
    Equation 19 from Wang+95
    All units in cgs

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    omega: float
        Fastness parameter
    M_NS: float
        Mass of the NS, defaults to 1.4 solar massess (in g)
    """
    N_0 = accretion_torque(Mdot, Rmag, M_NS)
    n = (7.0 / 6.0 - (4 / 3.0) * omega + (1 / 9.0) * omega**2.0) / (1.0 - omega)
    return N_0 * n


# if there are defaults do not type the function with jit, as it does not work with kwargs. We can use njit instead, but we cannot use the default value for M_NS (we can set it to 1.4 M_sun in grams, but it is not as clear)
@njit
def magnetic_torque_wang(Mdot, Rmag, omega, M_NS=M_NS_default, chi=0.0):
    """This is like the above, but only considering the magnetic term.
    It is a spin down term due to magnetic field lines threading the disc beyond the co-rotation radius

    Parameters
    ----------
    Mdot: float
        Instantaneous mass-accretion rate at Rmag. Units of g/s
    Rmag: float
        Magnetospheric radius in cm
    omega: float
        Fastness parameter
    M_NS: float
        Mass of the NS, defaults to 1.4 solar massess (in g)
    chi:float
        Magnetic obliquity angle in radians. Here the term cos(chi)**2 comes from the Bz component and the Bphi \propto Bz,
        which results in a cos(chi)**2 dependence. See Wang 1997
    """
    N_0 = accretion_torque(Mdot, Rmag, M_NS)
    return (
        N_0
        / (1.0 - omega)
        * (1.0 / 6.0 - omega / 3.0 + omega**2.0 / 9)
        * cos(chi) ** 2.0
    )


@njit
def accretion_torque_dai(
    Mdot: float,
    Rmag: float,
    omega: float,
    M_NS: float = M_NS_default,
    gamma: float = 1,
    delta: float = 0.1,
    psi: float = 0.5,
) -> float:
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
    xi = 2.0**0.5 * gamma * delta  # sqrt(2) * gamma * delta
    N_acc = (
        xi * accretion_torque(Mdot, Rmag, M_NS) * (1.0 - omega) / (psi**3.5)
    )  # e.g. Ghosh & Lamb 1979 Eq 2
    return N_acc


@njit
def magnetic_torque_dai(
    mu: float, Rmag: float, omega: float, gamma=1, chi: float = 0.0
) -> float:
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
    chi: float,
        Magnetic obliquity angle in radians. Here the term cos(chi)**2 comes from the Bz component and the Bphi \propto Bz,
        which results in a cos(chi)**2 dependence.
    """
    factor = gamma * mu**2.0 / (3.0 * Rmag**3)
    Nmag = factor * (1.0 - 2.0 * omega + 2.0 * omega**2.0 / 3.0) * cos(chi) ** 2.0
    return Nmag


@njit
def magnetic_torque_radial_twisting(
    mu: float,
    Rmag: float,
    omega: float,
    gamma: float = 1,
    h_0: float = 0.1,
    chi: float = 0.0,
) -> float:
    """Computes the torque due to radial shearing at R0 from Wang 1997 (Equation 8, but without the accretion and threaded disc contribution)

    h_0: float,
        The aspect ratio of the disk at Rmag (H/R).

    chi: float,
        Magnetic obliquity angle in radians. Here the term sin(chi)**2 comes from the Br component and the Bphi \propto Br, which results in a sin*(chi)**2 dependence.


    """
    factor = 2 * gamma * mu**2.0 / (Rmag**3.0)
    Nmag = factor * (1 - omega) * h_0 * sin(chi) ** 2.0
    return Nmag


@njit
def magnetic_torque_dai_propeller(
    mu: float, Rmag: float, omega: float, gamma: float = 1, chi: float = 0.0
):
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
    chi: float,
        Magnetic obliquity angle in radians. Here the term cos(chi)**2 comes from the Bz component and the Bphi \propto Bz,
        which results in a cos(chi)**2 dependence.
    """
    factor = gamma * mu**2.0 / (3.0 * Rmag**3.0)
    Nmag = factor * (2.0 / (3.0 * omega) - 1.0) * cos(chi) ** 2.0
    return Nmag


@jit(float64(float64, float64, float64, float64), nopython=True)
def propeller_torque(Mdot: float, M_NS: float, omega: float, Rm: float) -> float:
    """Equation 12 from Illarionov & Sunyaev 1975 or Eq 42 from Abolmasov 2024 review

    Parameters
    ----------
    omega: float,
        Angular velocity of the NS (2 pi / P)
    Rm :float,
        Magnetospheric radius
    """
    return -Mdot * Gcgs * M_NS / Rm / omega


@njit
def gravitational_quadrupole_torque(P: float, Q: float = 1e38):
    """Computes the gravitational quadrupole torque. See e.g. Equation 8 from Suvorov 2021.
    Parameters
    ----------
    P: float,
        Spin period of the NS in seconds
    Q: float, default 10^38
        Quadrupole moment of the NS in cgs units (default 10^38)
    Returns the gravitational quadrupole torque in erg/s
    """
    nu = 1 / P
    T_G = -(2**13) * Gcgs * pi**6 * Q**2 * nu**5 / (75 * ccgs**5)
    return T_G
