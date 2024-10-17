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

    @property
    def alpha(self):
        return self._alpha

    @property
    def omega(self):
        return self._omega

    @property
    def rco(self):
        return self._rco
    
    @property
    def chi(self):
        return self._chi

    @property
    def rlc(self):
        return self._rlc

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
        self._rco = self.corotation_radius() / self._Risco # in units of ISCO
        self._rlc = self.light_cylinder() / self._Risco # in units of ISCO
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
        return ((Gcgs *  self.M * self._P_NS**2 /(4 * np.pi**2)) ** (1/3))


    def light_cylinder(self):
        """Return the light cylinder (e.g. Biryukov and Abolmasov 2021)

        Parameters:
        -------
        P: float,
            Period of the NS in seconds

        Returns the light cylinder in cm
        """
        omega = 2 * np.pi / self._P_NS
        return ccgs / omega


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


def magnetospheric_radius(Mdot, B, M=(1.4 * M_sun).to(u.g).value, Rns=10**6, psi=0.5):
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
    return psi * ((Rns**12 * B**4) / (2 * Gcgs * M * Mdot**2)) ** (1/7)



def magnetospheric_radius_superEdd(NS, B, psi=0.5, sph_factor=5/3):
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


def torque_wang(Mdot, Rmag, Rco, M_NS=(1.4 * M_sun).to(u.g).value, xi=1):
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
    xi: float
        Dimensionless parameter of the order of unity
    """
    omega = fastness_parameter(Rmag, Rco)
    N_0 = Mdot * m.sqrt(Gcgs * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    n = (7/6 - (4/3) * omega + (1/9)*omega**2) / (1 - omega)
    return xi * N_0 * n


def torque(Mdot, Rmag, Rco, M_NS=1.4 * M_sun, xi=1):
    """Computes the accretion torque onto a NS

    Parameters
    ----------
    Mdot: astropy.quantity
        Instantaneous mass-accretion rate at Rmag. Units of mass/time
    Rmag: astropy.quantity
        Magnetospheric radius
    Rco: astropy.quantity
        Co-rotation radius
    M_NS: astropy.quantity
        Mass of the NS, defaults to 1.4 solar massess
    xi: float
        Dimensionless parameter of the order of unity
    """
    omega = fastness_parameter(Rmag, Rco)
    N_0 = Mdot.to(u.g / u.s) * np.sqrt(G * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    return (xi * N_0 * (1 - omega)).decompose(bases=u.cgs.bases)


def magnetic_moment(B, R_NS=10*u.km):
    """See e.g. Equation (2) from Tsygankov

    """
    return (B.to(u.G) * R_NS.to(u.cm)**3) / 2