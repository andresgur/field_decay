#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math as m
import astropy.units as u
from astropy.constants import G, c, M_sun


Gcgs = G.to(u.cm**3/u.g/u.s**2).value

ccgs = c.to(u.cm/u.s).value


class CO():
    """Base compact object class"""
    def __init__(self, M, spin=0):
        self.M = M
        self.Rg = gravitational_radius(self.M)
        self._spin = spin
        self._Risco = isco_radius(self.M, self._spin)
        efficiency = accretion_efficiency(self.M, self._Risco)
        self._Medd = eddington_luminosity((self.M * u.g).to(u.M_sun).value)/ ccgs** 2 / efficiency
    
    def __str__(self):
        return (f"Compact Object (CO):\n"
            f"Mass: {self.M:.2e} g\n"
            f"Spin: {self._spin}\n"
            f"Risco: {self._Risco:.2e} cm\n"
            f"Eddington mass-accretion rate: {self._Medd:.2e} erg/s")

    @property
    def Risco(self):
        return self._Risco

    @property
    def spin(self):
        return self._spin

    @property
    def Medd(self):
        return self._Medd

class NS(CO):
    """Neutron Star class"""
    def __init__(self, P_NS, Pso, M_NS=1.4 * u.M_sun, R_NS=10**6 * u.cm, chi=np.pi / 4, alpha=np.pi / 4):
        """
        P_NS: float,
            Spin period in seconds
        chi: float,
            Magnetic angle (measured from the spin axis; see e.g. Biryukov and Abolmasov 2021) in radiants
        alpha: float,
            Spin axis angle with respect to the disk (see e.g. Biryukov and Abolmasov 2021) in radiants
        """
        super().__init__(M_NS.to(u.g).value)
        self.R_NS = R_NS.to(u.cm).value
        self.I = self.moment_of_inertia()
        self.P_NS = P_NS #  this is the setter
        self.Pso = Pso * u.d
        self._chi = chi
        self._alpha = alpha

    def __str__(self):
        return (f"Neutron Star (NS):\n"
            f"Mass: {self.M:.2e} g\n"
            f"Radius: {self.R_NS:.2e} cm\n"
            f"Spin period: {self._P_NS:.2f} s\n"
            f"Spin: {self._spin}\n"
            f"Omega (angular velocity): {self._omega:.2e} rad/s\n"
            f"Risco: {self._Risco:.2e} cm\n"
            f"Rco (corotation radius cm): {self._Rco:.2f}\n"
            f"Rlc (light cylinder cm): {self._Rlc:.2f}\n"
            f"Eddington mass-accretion rate: {self._Medd:.2e} g/s")

    @property
    def alpha(self):
        return self._alpha

    @property
    def omega(self):
        return self._omega

    @property
    def Rco(self):
        return self._Rco
    
    @property
    def chi(self):
        return self._chi

    @property
    def Rlc(self):
        return self._Rlc

    @property
    def P_NS(self):
        return self._P_NS

    @P_NS.setter
    def P_NS(self, period):
        """Set the period of the NS (in seconds)"""
        self._P_NS = period
        self._omega = 2 * np.pi / period # angular velocity
        self._spin = self.period_to_spin()
        self._Risco = isco_radius(self.M, self._spin)
        self._Rco = self.corotation_radius() # in units of ISCO
        self._Rlc= self.light_cylinder() # in units of ISCO
        efficiency = accretion_efficiency(self.M, self._Risco)
        self._Medd = eddington_luminosity((self.M * u.g).to(u.M_sun).value)/ ccgs** 2 / efficiency
        
    
    def corotation_radius(self,):
        """Return the corotation radius (e.g. equation (1) from Tysgankov et al 2016). Everything in cgs

        Parameters:
        -------
        P:float,
            Period of the NS in s
        M_NS:float
            Mass of the NS in g

        Returns the corotation radius in cm
        """
        return (Gcgs *  self.M / self._omega**2) ** (1/3)


    def light_cylinder(self):
        """Return the light cylinder (e.g. Biryukov and Abolmasov 2021)

        Parameters:
        -------
        P: float,
            Period of the NS in seconds

        Returns the light cylinder in cm
        """
        return ccgs / self.omega


    def moment_of_inertia(self,):
        """Moment of Inertia of a NS (see http://hyperphysics.phy-astr.gsu.edu/hbase/isph.html). All units in cgs
        Returns the moment of inertia in cgs
        """
        return (2 * self.M * self.R_NS**2 / 5)

    def spin_to_period(self, a_spin):
        """Convert spin of a NS to period
        
        Returns the period in seconds
        """
        P = 2 * np.pi * self.I / a_spin / Gcgs / self.M**2
        return P

    def period_to_spin(self):
        """Convert Period to spin of the NS"""
        a_spin = 2 * np.pi * self.I * ccgs / self._P_NS / Gcgs / self.M**2
        return a_spin


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
    N_0 = Mdot * m.sqrt(Gcgs * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
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
    return N_0 / (1 - omega) * (1/6 -omega/3 + (omega**2)/9)


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