#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math as m
import astropy.units as u
from astropy.constants import G, c, M_sun, R_sun, L_sun, sigma_T, m_p


class CO():
    """Base compact object class"""
    def __init__(self, M, spin=0):
        self.M = M
        self.Rg = gravitational_radius(self.M)
        self._spin = spin
        self._Risco = isco_radius(self.M, self._spin).to(u.cm)
        efficiency = accretion_efficiency(self.M, self._Risco)
        self._Medd = eddington_luminosity(self.M).decompose(bases=u.cgs.bases) / (c.to(u.cm/u.s))** 2 / efficiency

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
    def __init__(self, P_NS, Pso, M=1.4 * u.M_sun, R=10**6 * u.cm):
        super().__init__(M)
        self.R_NS = R
        self.I = moment_of_inertia(self.M, self.R_NS)
        self._P_NS = P_NS * u.s
        self.Pso = Pso * u.d
        self._spin = period_to_spin(self._P_NS, self.M, self.R_NS)
        self._Risco = isco_radius(self.M, self._spin).to(u.cm)
        self._Rco = corotation_radius(self._P_NS).to(u.cm) / self._Risco # in units of ISCO
        efficiency = accretion_efficiency(self.M, self._Risco)
        self._Medd = eddington_luminosity(self.M).decompose(bases=[u.cm, u.g, u.s]) / (c.to(u.cm/u.s))** 2 / efficiency

    @property
    def Rco(self):
        return self._Rco


    @property
    def P_NS(self):
        return self._P_NS

    @P_NS.setter
    def P_NS(self, period):
        """Set the period of the NS (in seconds)"""
        self._P_NS = period * u.s
        self._spin = period_to_spin(self._P_NS, self.M, self.R_NS)
        self._Risco = isco_radius(self.M, self._spin).to(u.cm)
        self._Rco = corotation_radius(self._P_NS).to(u.cm) / self._Risco # in units of ISCO
        efficiency = accretion_efficiency(self.M, self._Risco)
        self._Medd = eddington_luminosity(self.M).decompose(bases=[u.cm, u.g, u.s]) / (c.to(u.cm/u.s))** 2 / efficiency


def accretion_efficiency(M, R):
    """Returns the accretion efficiency.
    M: astropy.quantity
        Mass of the compact object in kg
    R: astropy.quantity
        Radius of the compact object or innermost stable orbit
    Returns the accretion efficiency
    """
    return G.to(u.cm**3/u.g/u.s**2) * M.to(u.g)  / (2 * c.to(u.cm/u.s) ** 2 * R.to(u.cm))


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
    efficiency = accretion_efficiency(M, R)
    return efficiency * M_dot * c.to(u.cm/u.s) ** 2


def magnetospheric_radius(Mdot, B, M=1.4 * u.M_sun, Rns=10**6 * u.cm, psi=0.5):
    """Return the magnetospheric radius given by my thesis (dipole magnetic field) Equation 2.19

        Parameters:
        -----------
        Mdot: astropy.quantity
            Mass accretion rate
        B: astropy.quantity
            Magnetic field
        M: astropy.quantity,
            Mass of the NS
        Rns: astropy.quantity,
            Radius of the NS
        psi:float
            Geometrical factor
        Returns the magnetospheric radius
    """
    B = B.to(u.G).value * (u.cm**-0.5 * u.g**0.5 / u.s)
    return psi * ((Rns.to(u.cm)**12 * B**4) / (2 * G.to(u.cm**3 /u.g/u.s**2) * M.to(u.g) * Mdot.to(u.g/u.s)**2)) ** (1/7)


def magnetospheric_radius_B(R, L_39, psi=0.5, m=1, R_6=1):
    """Return the magnetic field given by the radius in km and the luminosity in 10^39 erg/s. Equation 22 from Mushtukov

        Parameters:
        -----------
        R: float,
            Inner disk radius in km
        L_39: float,
            The accretion luminosity in 10³⁹ erg/s
        psi:float
            That factor
        m: float
            Mass of the NS in M_sun
        R_6: float
            Radius of the NS in 10⁶ cm

        Returns the magnetic field in 10¹² G.
    """
    factor = 7 * 10 ** 7 * psi * m ** (1.0 / 7.0) * R_6 ** (10.0 / 7.0) * L_39 ** (-2.0/7.0) * u.cm
    B = (R.to(u.cm) / factor) ** (7.0 / 4.0)
    return B


def spherization_radius_poutanen(m_0, e_wind=0.5):
    """Equation 21 from Poutanen et al 2007

        Parameters
        ----------
        m_0: float
            Mass-transfer rate at the donor in Eddington units
        e_wind: float, default 0.5
            Fraction of radiative energy that goes to accelerate the outflow
        Returns Rsph in units of Rin (see around Equation 7)
        """
    return (1.34 - 0.4 * e_wind + 0.1 * e_wind ** 2 - (1.1 - 0.7 * e_wind) * m_0 ** (-2/3)) * m_0


def optical_depth_perpen(m_0, m_in, rsph, beta = 1, r=10):
    """Equation 29 from Poutanen et al 2007

        Parameters
        ----------
        m_0: float
            Mass-transfer rate at the donor
        m_in: float
            Mass-transfer rate at the inner radius
        rsph: float
            Spherization radius in units of m_0
        beta: float
            Not very clear from Poutanen but gives the maximum velocity the outflow reaches asymptotycally, default 1 as in the paper
        r: float
            The radius at which the optical depth wants to be computed (in Rin units)
        Returns the optical depth through the outflow in the perpandicular direction at the given radius (valid only inside the spherization radius)
        """
    if r<=rsph:
        factor = (r ** 0.5 - r ** -0.5 )
    else:
        factor = (r_sph - 1) * r_sph ** 0.5 * r ** -1
    return factor * 5 / beta * (m_0 - m_in) / rsph


def super_edd_luminosity_noadvection(m_0, r_sph, r):
    """Equation 16 from Poutanen et al 2007

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor
    r_sph: float
        Spherization radius in units of m_0
       r: float, single element
        The radius at which the luminosity wishes to be computed (beyond or below Rsph)
    Returns the luminosity below or beyond Rsph in Eddington ratio
    """
    factor = m_0 / r_sph / (1 + 2/3 * r_sph **-5/2)
    if r <= r_sph:
        factor = np.log(r_sph) - 2/5 * (1 - r_sph**(-5/2))
    else:
        factor = 5/3
    return m_0 / r_sph / (1 + 2/3 * r_sph **-5/2) * factor


def super_edd_luminosity_shakura(m_0):
    """Equation 19 from Poutanen et al 2007 (see Shakura & Sunyaev 1973 model)

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor
    Returns the luminosity below or beyond Rsph in Eddington ratio
    """
    return (1 + 3/5 *np.log(m_0))

def neutron_star_binding_luminosity(Mdotmag, Rmag, M_NS=1.4 * u.M_sun, R_NS=10**6 * u.cm):
    """Computes the luminosity within Rmag (See Middleton+2022 Equation 2)

    Mdotmag: astropy.quantity
        Mass-accretion rate at the magnetospheric radius
    """
    L_NS = (G * M_NS * Mdotmag).decompose(bases=[u.cm, u.s, u.g]) * (1 / R_NS - 1 / Rmag)
    return L_NS

def luminosity_super_edd_NS(mdot, B, spin=0, e_wind=0.5, psi=1, M_NS=1.4 * u.M_sun, R_NS=10**6 * u.cm):
    """Equations from Middleton+22. Only works in the case where Rsph > Rmag
    mdot: float
        Mass-transfer rate in dimensionless units
    B:astropy.quantity
        Magnetic field dipole
    e_wind: float
        Energy imparted to the wind
     """
    Risco = isco_radius(M_NS, spin).to(u.cm)
    Rsph = spherization_radius_poutanen(mdot, e_wind) * Risco
    Rg = gravitational_radius(M_NS).to(u.cm)
    efficiency = 1 / (2 * Risco / Rg)
    Ledd = eddington_luminosity(M_NS).decompose(bases=[u.cm, u.s, u.g])
    Medd = Ledd / (c.to(u.cm/u.s))** 2 / efficiency
    results = mass_transfer_rate_mag_radius(mdot, B, M_NS, a_spin=spin, psi=psi, e_wind=e_wind)
    Rmag = results[2].to(u.cm)
    mdotmag = results[0]
    L = Ledd # contribution from the outer disk

    # contribution from the region within Rmag
    if Rmag < Risco:
        Rmag = Risco
    L_NS = neutron_star_binding_luminosity(mdotmag, Rmag, M_NS, R_NS)
    L_NS = (G * M_NS * mdotmag).decompose(bases=[u.cm, u.s, u.g]) * (1 / R_NS - 1 / Rmag)
    L = L + L_NS
    # region from Rin to Rsph
    if Rsph > Rmag:
        L = L + Ledd * np.log(Rsph / Rmag) * (1 - e_wind)
    return L


def mass_transfer_noadvection(mdot, Rin, R):
    """Returns Equation 19 from Lipunova+99 (See also Poutanen without advection)
    Mdot: float
        Mass-transfer at the outer radius in Eddington ratio
    Rin: astropy.quantity
        Inner most disk radius (ISCO or magnetospheric radius)
    R: astropy.quantity
        Radius at which to compute the mass transfer rate
    """
    R = R.to(u.cm)
    if len(np.atleast_1d(R))==1:
        R = np.array([R.to(u.cm).value]) * u.cm
    m_r = np.empty(len(R))
    R_sph = 1.62 * mdot * Rin.to(u.cm)
    m_r[R<=R_sph] = mdot * (R_sph/R[R<=R_sph])**(3 / 2) * (1 + 3/2 *(R[R<=R_sph]/Rin)**(5/2)) / (1 + 3/2 * (R_sph / Rin)**(5/2))
    m_r[R>R_sph] = mdot
    return m_r


def radiative_energy_noadvection(mdot, M, Rin, R):
    """Returns Equation 14 from Poutanen+2007 from the Lipunova+99 model in units of g_0 (see Poutanen + 2007)
    Mdot: float
        Mass-transfer at the outer radius in Eddington ratio
    R: astropy.quantity
        Radius
    """
    def g_r(mdot, r, rsph):
        return mdot / 3 * r ** (3/2) / r_sph * (1 - r ** (-5 /2)) / (1 + 2/3 * r_sph ** (-5/2))
    # calculate g_0
    kt = sigma_T.to(u.cm**2) / m_p.to(u.g)
    G_cgs = G.decompose(bases=[u.cm, u.g, u.s])
    Medd = 48 * np.pi * G_cgs * M.to(u.g) / c.to(u.cm/u.s) / kt #as defined in Poutanen
    g_0 = Medd * np.sqrt(G_cgs * M.to(u.g) * Rin.to(u.cm))

    g_rad = np.empty(len(R))
    r_sph = 5/3 * mdot * Rin.to(u.cm) / Rin.to(u.cm)
    r = R.to(u.cm) / Rin.to(u.cm)
    g_rad[r<=r_sph] = g_r(mdot, r[r<=r_sph], r_sph)
    g_rad[r>r_sph] = g_r(mdot, r_sph, r_sph) + mdot * (r[r>r_sph]**0.5  - r_sph **0.5)
    Qrad = 3 / 8 / np.pi * (G_cgs * M.to(u.g))**0.5 * (g_rad * g_0) / R.to(u.cm)**(7/2)
    return Qrad


def radiative_energy(m_0, M, Risco, R, e_wind=0.5):
    """Returns Equation 12 from Lipunova+1999 using Poutanens mass transfer rate with advection
    m_0: float
        Mass-transfer at the outer radius in Eddington ratio
    R: astropy.quantity
        Radius
    """
    kt = sigma_T.to(u.cm**2) / m_p.to(u.g)
    G_cgs = G.decompose(bases=[u.cm, u.g, u.s])
    Medd = 48 * np.pi * G_cgs * M.to(u.g) / c.to(u.cm/u.s) / kt #as defined in Poutanen
    w = keplerian_angular_w(R, M)
    min = mass_transfer_inner_radius(m_0, e_wind)
    R_sph = spherization_radius_poutanen(m_0, e_wind) * Risco
    dm_dr = np.empty(len(R))
    dm_dr[R<=R_sph] = (m_0 - min) / R_sph.to(u.cm) * Medd
    dm_dr[R>R_sph] = m_0 * Medd * (1 - (R/Risco)**0.5) / R.to(u.cm) #SS73 solution
    Qrad = w**2 * R.to(u.cm) / 8 / np.pi * dm_dr
    return Qrad

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

    return (1 - a) / (1 - a * (2/5 * m_0) ** (- 0.5))

def mass_transfer_rate_mag_radius(mdot, B=10**12*u.G, M_NS=1.4 * M_sun, R_NS=10**6*u.cm, psi=0.5, a_spin=0.3, e_wind=0.5):
    """Equation (6) from Mushtukov et al. 2019, where R is replaced by Rmag
    Returns the mass-transfer rate at the magnetospheric radius by solving Equation (6) numerically in physical units and the magnetospheric radius
    """
    B = B.to(u.G)
    isco = isco_radius(M_NS, a_spin).to(u.cm)
    efficiency = 1 /  (2 * isco / gravitational_radius(M_NS).to(u.cm))
    eddington_rate = eddington_luminosity(M_NS).decompose(bases=[u.cm, u.g, u.s]) / (c.to(u.cm/u.s))** 2 / efficiency
    #eddington_rate = eddington_accretion_rate(M_NS, R_NS)
    M_0 = eddington_rate * mdot
    M_isco = mass_transfer_inner_radius(mdot, e_wind) * M_0
    # we effectively assume isco = R_NS
    r_sph = spherization_radius_poutanen(mdot, e_wind) * isco
    mass_acc_a = M_0 * 0.01
    mass_acc_b = M_0 * 0.99
    err_tol = M_0 * 10 ** -5
    err = (mass_acc_b - mass_acc_a) / 2

    while err > err_tol:
        mass_acc = (mass_acc_a + mass_acc_b) / 2
        #rmag = magnetospheric_radius(mass_acc, B, M_NS, R_NS, psi=psi).to(u.cm)
        rmag = magnetospheric_radius_ls(mass_acc, B, M_NS, R_NS).to(u.cm)
        m_r = M_isco + (M_0 - M_isco) * rmag / r_sph
        f_mass_acc = mass_acc - m_r
        err = np.abs(mass_acc_b - mass_acc_a) / 2
        #rmag = magnetospheric_radius(mass_acc_a, B, M_NS, R_NS, psi=psi).to(u.cm)
        rmag = magnetospheric_radius_ls(mass_acc_a, B, M_NS, R_NS).to(u.cm)
        m_r = M_isco + (M_0 - M_isco) * rmag / r_sph
        f_mass_acc_a = mass_acc_a - m_r
        if f_mass_acc_a * f_mass_acc <0:
            mass_acc_b = mass_acc
        else:
            mass_acc_a = mass_acc

    return mass_acc, err, magnetospheric_radius_ls(mass_acc, B, M_NS, R_NS) # , psi=psi


def spherization_temperature(M, m_0, e_wind=0.5):
    """From Poutanen 2007. Equation 37.

        Parameters
        ----------
        M: in solar units
        m_0: float
            Mass-transfer rate at the donor
        e_wind: 0.5
            Between 0 and 1. Energy fraction that goes into the wind
        Returns the temperature at the spherization radis in keV
        """
    return 1.5 * (M)**(-1/4) * m_0 **(-1/2) * (1 + 0.3 * m_0**(-3/4)) * (1 - e_wind)**(1/4)


def find_mdot(T_sph, M, e_wind=0.5, fcol=1.5, a_spin=0):
    """Inver Equation (37) from Poutanen et al. 2007, modified by Middleton+19 Equation (19) to take into account the R_isco,
     in order to infer the mass-transfer rate
    T_sph: astropy.quantity
        Temperature at the spherization radius
    Returns the mass-transfer rate in dimensionless units
    """
    R_isco = isco_radius(M, a_spin)
    R_g = gravitational_radius(M)
    r_isco = R_isco.to(u.km) / R_g.to(u.km)

    mdot = (1.5 * fcol * M.to(M_sun).value**(-1/4) * (6 /r_isco)**0.5 * (1 -e_wind)**(1/4) / T_sph.to(u.keV).value)**2
    return mdot


def photosphere_temperature_ls(m_dot, M=10 * M_sun, e_wind=0.5, fcol=1.7, a_spin=0.998):
    """From Middleton 2019 2007. Equation 19.
        Parameters
        ----------
        M: Quantity
            in solar units
        m_dot: float
            Mass-transfer rate at the donor
        e_wind: float
            Between 0 and 1. Energy fraction that goes into the wind
        Returns the temperature of the photosphere of the outflow in keV
    """
    isco = isco_radius(M, a_spin).to(u.cm) / gravitational_radius(M).to(u.cm)
    return 1.5 * fcol * m_dot **(-1/2) * (M.to(u.M_sun).value) **(-1/4) * (1 - e_wind)**(1/4) * (6 / isco) ** (1/2)


def photosphere_temperature(M, m_dot, beta=1.4, chi=1, e_wind=1/2):
    """From Poutanen 2007. Equation 38.
        Parameters
        ----------
        M: in solar units
        m_0: float
            Mass-transfer rate at the donor
        beta: float, optional default =1
            Not very clear from Poutanen but gives the maximum velocity the outflow reaches asymptotycally, default 1 as in the paper)
        Returns the temperature of the photosphere of the outflow in keV
        """
    return 0.8 * (beta * chi / e_wind)** 1/2 * M**(-1/4) * m_0 **(-3/4)


def disk_temperature(M, m_dot):
    """From Poutanen 2007. Equation 36.
        Parameters
        ----------
        M: in solar units
        m_0: float
            Mass-transfer rate at the donor
        Returns the temperature of the inner disk in keV
        """
    return 1.6 * M ** (-1/4) * (1 - 0.2 * m_0 ** (-1/3))


def isco_radius(M, a=0.998):
    """Returns the ISCO radius for a given mass in km.
        Parameters
        ----------
        M: astropy.quantity
        a: float,
        dimensionless spin: 0 for a Scharzschild black hole or 0.998 for a Kerr black hole."""
    z1 = 1 + (1 - a**2) ** (1/3) * ((1 + a)** (1/3) + (1-a) ** (1/3))
    z2 = np.sqrt(3 * a ** 2 + z1**2)
    return (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * gravitational_radius(M)


def schwarzschild_radius(M):
    """Returns the scharzchild radius for a given mass in km.
        Parameters
        ----------
        M: astropy.quantity
    """
    return 2 * gravitational_radius(M).to(u.cm)


def gravitational_radius(M):
    """Returns the scharzchild radius for a given mass in km.
    Parameters
    ----------
    M: astropy.quantity
        Mass of the compact object
    """
    return (G * M.to(u.kg)/ c**2).decompose(bases=[u.cm])


def eddington_luminosity(M):
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: astropy.quantity
        Returns the Eddington luminosity
    """
    return 1.26 * M.to(u.M_sun).value * 10**38 * u.erg/u.s


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


def mushtukov_envelope_Bfield(Tenvelope, L, R_NS=1, M_NS=1.4):
    """From Mushtukov et al 2017 (10). Returns the magnetic field value given by enevolpe temperature.

        Parameters:
        -----------
        T: float,
            Temperature of the hot component in keV units
        L:float
            Total luminosity in units of 10^39 erg/s
        Returns the magnetic field value in units of 10^12 G"""

    # R_NS is dimensionless in Mushtukov equations
    R_NS = 1
    M_NS = 1.4
    # in units of 10**12 G
    B = (2 * Tenvelope * L ** (-11 / 28) * M_NS ** (1 / 14) * (R_NS) ** (5/7)) **(- 7 /2)
    return B


def mushtukov_disk_Bfield(Tdisk, L, R_NS=1, M_NS=1.4):
    """From Mushtukov et al 2017 (11). Returns the magnetic field value given by the disk temperature.

        Parameters:
        -----------
        Tdisk: float,
            Temperature of the hot component in keV units
        L:float
            Total luminosity in units of 10^39 erg/s
        Returns the magnetic field value in units of 10^12 G"""

    # R_NS is dimensionless in Mushtukov equations
    R_NS = 1
    M_NS = 1.4
    # in units of 10**12 G
    B = (4 * Tdisk * L ** (-13 / 28) * M_NS ** (3 / 28) * (R_NS) ** (23/28)) **(- 7 /3)
    return B

def Pmag(B, Rin, R_NS=10**6 * u.cm):
    "Magnetic pressure"
    mu = B.to(u.G).value * (u.cm ** (-0.5) * u.g**(0.5) /u.s) * R_NS ** 3
    return (mu**2 / (8 * np.pi * Rin**6)).decompose(bases=u.cgs.bases)


def Prad(L, Rin):
    """Radiative pressure"""
    return (L / (4 * np.pi * Rin**2 * c)).decompose(bases=u.cgs.bases)

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


def disknorm_tosize(norm, distance, angle=0):
    """Returns the disk radius in km from its normalization. (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node164.html)

        Parameters:
        ----------
        norm:float,
            The norm of the disk
        distance: float
            The distance of the source in megaparsecs
        angle: float
            The inclination angle of the disk
    """
    angle_rad = angle / 360 * 2 * m.pi
    # mega parsecs to parsecs
    distance = distance.to(u.kpc)
    r_in = np.sqrt(norm / m.cos(angle_rad)) * (distance.value / (10)) * u.km
    return r_in


def radius_tomass(r_in, spin="schw"):
    if spin=="schw":
        return r_in * (c.to("km/s")) ** 2 / 6 / G.to("km**3/kg/s**2") / M_sun
    elif spin=="kerr":
        return r_in * (c.to("km/s")) ** 2 / G.to("km**3/kg/s**2") / M_sun
    else:
        print("Error: Spin option %s not valid, valid options are schw or kerr")
        return 0

def ratio_luminosities_propeller(P, M_NS=1, R_NS=1):
    """Return the ratio of minimum and maximum luminosities due to the propeller effec (e.g. equation (5) from Tysgankov et al 2016)

        Parameters:
        -------
        P: float,
            Period in seconds
        M_NS:float
            Mass of the NS in units of 1.4 M_sun
        R_NS: float
            Radius of the NS in units of 10⁶ cm
        Returns the ratio of the observed luminosities
        """
    return 170 * P** (2 / 3) * M_NS ** (1 / 3) * R_NS ** -1


def corotation_radius(P, M_NS=1.4 * u.M_sun):
    """Return the corotation radius (e.g. equation (1) from Tysgankov et al 2016)

    Parameters:
    -------
    P: astropy.unit,
        Period of the NS
    M_NS:astropy.unit
        Mass of the NS
    """
    return ((G *  M_NS * P**2 /(4 * np.pi**2)) ** (1/3)).decompose(bases=u.cgs.bases)


def b_propeller(luminosity, period, psi = 0.5, R_NS=1, M_NS=1.4):
    """Returns the maximum magnetic field a source can have given its minimum luminosity so it does not enter the propeller regime (equation 37 Mushtukov et al 2015)
        Parameters:
        luminosity: float
            In units of 10^39 erg/s
        period:float
            Period in seconds
        psi: float
            0.5 for disk, 1 for spherical geometry
        R_NS: float
            In units of 10⁶ cm
        M_NS: float
            In units of M_sun
        Returns the magnetic field in 10^12 G
    """
    factor = 7.0 * 10.0 ** (-2.0) * psi ** (7.0 / 2.0) * M_NS **(-2.0/3.0) * R_NS ** (5.0) * period ** (-7.0/3.0)
    B = (luminosity / factor) ** (0.5)
    return B



def error_mean(values, uppererrors, lowerrors):
    """Compute the mean and the error based on the individual errors of each value. See http://avntraining.hartrao.ac.za/images/Error_Analysis.pdf page 7

        Parameters:
        ----------
        values: array
            The values
        uppererrors: array
            The upper limits of each of the values
        lowerrors: array
            The lower limits of each of the values

        Returns the mean and its error.
        """
    mean = np.mean(values)
    error_up = np.sqrt(np.sum([x ** 2 for x in uppererrors])) / len(values)
    error_down = np.sqrt(np.sum([x ** 2 for x in lowerrors])) / len(values)
    return mean, error_up, error_down

def orbital_separation(period, M_acc, M_star):
    """Derive orbital separation using Keppler's law.
    Parameters:
    -----------
    period: in days
    M_acc: mass of the accretor in solar masses
    M_star: mass of the companion star in solar masses

    Returns the orbital separation in solar radius.
    """
    d = ((period ** 2) * G.to(u.km**3/u.g/u.d**2) * (M_acc + M_star)* M_sun.to(u.g) / (4 * np.pi**2)) ** ( 1 / 3)
    return d / R_sun.to(u.km)


def roche_lobe(orbital_separation, M_acc, M_star):
    """Compute the Roche Lobe overflow.
    Parameters:
    -----------
    orbital_separation: float
        Orbital separation in units of distance
    M_acc: mass of the accretor in solar masses
    M_star: mass of the companion star in solar masses

    Returns the Roche Lobe overflow of the donor in the same units as the orbital_separation
    """
    q = M_star / M_acc
    R = orbital_separation * 0.49 * q ** (2/3) / (0.6 * q ** (2/3) + np.log(1 + q**(1/3)))
    return R


def r_out(mdot, m_in, r_isco, r_sph, r_in=None, beta=1.4):
    """Outer photospheric radius of the wind. All radii must be given in gravitational radii (i.e. dimensionless)

    mdot: float
        Mass-transfer rate at the companion in Eddington units
    m_in: float
         Mass-accretion rate at the inner disk radius (ALWAYS isco, even for magnetised NSs)

    Returns the outer photospheric radius in units of Rg
    """

    if r_in is None:
        r_in = r_isco

    optical_depth = 5 * np.sqrt(r_isco / 6)
    rout = optical_depth / beta * (mdot - m_in) / np.sqrt(r_sph / r_isco)  * (r_sph / r_isco - r_in / r_isco) * r_isco
    return rout


def p_wind(mdot, m_in, a_spin=0.998, M=20 * u.M_sun, rin=None, beta=1.4, e_wind=0.25):
    """Compute the precessing period of the wind based on the Lense-Thirring effect (Equation 3 from Middleton et al. 2019).
    Parameters:
    -----------
    mdot: float
        Mass-transfer rate at the companion in Eddington units
    m_in: float
        Mass-accretion rate at the inner disk radius (ALWAYS isco, even for magnetised NSs)
    a_spin: float (optional, 0.998 by default)
            Dimensionless spin of the compact object
    M: float
        Mass of the compact object in solar masses
    rin: astropy.quantity
        Inner radius of the disk, magnetospheric radius for a NS. If None assume Risco as the inner radius
    Returns the precession period of the wind in astropy.quantity (days)
    """
    # gravitational radius
    rg = gravitational_radius(M)
    r_isco = isco_radius(M, a_spin).to(u.cm) / rg.to(u.cm)
    # assume inner disk radius = isco
    if rin is None:
        rin = r_isco
    else:
        rin = rin.to(u.cm) / rg.to(u.cm)
    optical_depth = 5 * np.sqrt(r_isco / 6)

    r_sph = spherization_radius_poutanen(mdot, e_wind) * r_isco

    if r_sph > rin:
        rout = optical_depth / beta * (mdot - m_in) / r_sph  * (r_sph - rin) * r_isco * np.sqrt(r_sph)
        denominator = 3 * (c.to(u.km / u.d) ** 3) *  a_spin
        #print("mdot: %.2f" % mdot)
        #print("min: %.2f" % m_in)
        #print("rout: %.2f" % rout)
        #print("rin: %.2f" % rin)
        #print("risco: %.2f" % r_isco)
        #print("rsph: %.2f" % r_sph)
        factor =  (1 - (rin / rout)**3) / np.log(rout / rin)
        P =  G.to(u.km**3 / u.g / u.d ** 2) * M.to(u.g) * np.pi * (rout ** 3) * factor / denominator
        return P, rout
    else:
        return np.nan * u.d


def p_wind_matt(mdot, m_in, a_spin=0.998, M=20 * u.M_sun, rin=None, beta=1.4, e_wind=0.25):
    """Compute the precessing period of the wind based on the Lense-Thirring effect (Equation 3 from Middleton et al. 2019). Here the radii in Rout are rescaled by Risco
    Parameters:
    -----------
    mdot: float
        Mass-transfer rate at the companion in Eddington units
    m_in: float
        Mass-accretion rate at the inner disk radius (isco or magnetospheric radius for magnetised NSs)
    a_spin: float (optional, 0.998 by default)
            Dimensionless spin of the compact object
    M: float
        Mass of the compact object in solar masses
    rin: astropy.quantity
        Inner radius of the disk, magnetospheric radius for a NS. If None assume Risco as the inner radius
    Returns the precession period of the wind in astropy.quantity (days)
    """
    # gravitational radius
    rg = gravitational_radius(M).to(u.cm)
    r_isco = isco_radius(M, a_spin).to(u.cm) / rg # in Rg
    # assume inner disk radius = isco
    if rin is None:
        rin = r_isco
    else:
        rin = rin.to(u.cm) / rg
    optical_depth = 5 * np.sqrt(r_isco / 6)

    r_sph = spherization_radius_poutanen(mdot, e_wind) * r_isco # in R_g

    if r_sph > rin:
        rout = optical_depth / beta * (mdot - m_in) / np.sqrt(r_sph / r_isco)   * (r_sph - rin) # r_sph - rin / risco x risco =  r_sph - rin
        denominator = 3 * (c.to(u.km / u.d) ** 3) *  a_spin
        #print("mdot: %.2f" % mdot)
        #print("min: %.2f" % m_in)
        #print("rout: %.2f" % rout)
        #print("rin: %.2f" % rin)
        #print("risco: %.2f" % r_isco)
        #print("rsph: %.2f" % r_sph)
        factor =  (1 - (rin / rout)**3) / np.log(rout / rin)
        P =  G.to(u.km**3 / u.g / u.d ** 2) * M.to(u.g) * np.pi * (rout ** 3) * factor / denominator
        return P, rout
    else:
        return np.nan * u.d, np.nan


def magnetospheric_radius_ls(Mdot, B, M_NS=1.4 * M_sun, R_NS=10**6 * u.cm):
    """As given by Middleton+19 accounts for super Edd accretion but no advection"""
    Mdot_17 = Mdot.to(u.g/u.s).value / 10**17
    mu = B.to(u.G) * R_NS.to(u.cm)**3 / (10**30 * u.G * u.cm**3)
    rm = 2.9 * 10**8 * Mdot_17**(-2/7) * (M_NS.to(u.M_sun).value)**(-1/7) * mu**(4/7) * u.cm
    return rm

def moment_of_inertia(M=1.4 * M_sun, R=10**6 * u.cm):
    """Moment of Inertia of a NS (see http://hyperphysics.phy-astr.gsu.edu/hbase/isph.html)"""
    return (2 * M * R**2 / 5).decompose(bases=[u.g, u.cm])

def spin_to_period(a_spin, M=1.4 * M_sun, R=10**6 * u.cm):
    """Convert spin of a NS to period"""
    I = moment_of_inertia(M, R)
    P = 2 * np.pi * I  * c.to(u.cm / u.s)/ a_spin / G / M**2
    return P.decompose(bases=u.cgs.bases)

def period_to_spin(period, M=1.4 * M_sun, R=10**6 * u.cm):
    """Convert Period to spin of the NS"""
    I = moment_of_inertia(M, R)
    a_spin = 2 * np.pi * I.to(u.g * u.cm**2) * c.to(u.cm / u.s)/ period.to(u.s) / G.decompose(bases=[u.cm, u.g, u.s]) / M.to(u.g)**2
    return a_spin


def magnetic_precession(R_mag, B=10**12 * u.G, R_NS =10**6 *u.cm, P_NS=1 * u.s, I_NS = 10**45 * u.g * u.cm**2,
                        spin_angle=30/360 * 2 *np.pi, mag_angle=30/360 * 2 *np.pi):
    """Equation 13 from Vasilopoulos2020 (Lipunov+1990)"""
    # define dimensionless units
    B = B.to(u.G) / (10**12 * u.G)
    R_NS = R_NS.to(u.cm) / (10**6 * u.cm)
    R_mag = R_mag.to(u.cm) / (10**8 * u.cm)
    P_NS = P_NS.to(u.s) / (1 * u.s)
    I_NS = I_NS.to(u.g * u.cm**2) / (10**45 * u.g * u.cm**2)
    P_mag = 1.5 * 10** 4 * B**-2 * R_NS**-2 * R_mag**3 * P_NS **-1 * I_NS / (np.cos(spin_angle) * (3 * np.cos(mag_angle)**2 - 1)) * u.yr
    return P_mag


def lense_thirring_magnetic_torque(R_in, R_out, surface_density, spin, B=10**12 * u.G, M_NS=1.4 * u.M_sun, R_NS=10**6 * u.cm, beta=1.4):
    """Equation 10 from Middleton+2019
    The radii need to be in physical units, even if the Equation says otherwise. Otherwise the unis do not match
    """
    Bin = (B.to(u.G).value * (u.cm**-0.5 * u.g**0.5 / u.s)/2 * (R_NS / R_in)**3).decompose(bases=u.cgs.bases)
    r_in = R_in.to(u.cm)
    r_out = R_out.to(u.cm)
    GM = (G * M_NS).decompose(bases=u.cgs.bases)
    ratio_angles = 1 # *np.tan(delta) / np.sin(beta)
    # Swap Rin by Rout from the original formula to get the correct sign!
    ratio = Bin**2  * r_in**6 * (r_in**-3 - r_out**-3) * ratio_angles / (48 * np.pi**2 * spin * (GM) ** (5/2) * c.to(u.cm/u.s)**-3 * surface_density * r_in **(-0.5) *np.log(r_out / r_in))
    return ratio

def kozai_period(P_orb, P_sup, K=1):
    """Returns the expected period based on the orbital and superorbital periods.
    Equation (19) from Zdziarski+2007

    Paramters
    P_orb: astropy.quantity
        The orbital period of the system
    P_sup: astropy.quanityt
        The superorbital period
    K: float
        Normally 1
    """
    P2 = np.sqrt(P_sup.to(u.yr) * P_orb.to(u.yr) / K)
    return P2.to(u.d)

def magnetic_factor(R, Rmag):
    """Magnetic factor taking into account the magnetic field threadening the disk. Equation A11 from Lai+1999
    R: astropy.quantity
        The radius at which the factor is to be calculated
    Rm: astropy.quantity
        The magnetospheric radius
    """
    F = 1 - 1/6 * (Rmag/R)**0.5 * (7 - (Rmag/R)**3)
    return F

def keplerian_angular_w(R, M=1.4 * M_sun):
    """Calulate the Keplerian angular velocity at a given radius"""

    return np.sqrt(G.decompose(bases=[u.cm, u.g, u.s]) * M.to(u.g) / R.to(u.cm)**3)


def surface_density(R, Mdot, Risco, M=1.4 * M_sun, alpha=0.1):
    """Returns the surface density of the radiation pressure dominated disk (Equation 2.9 from Shakura & Sunyaev 1973)
    Tested against Equation 6.8 from Lai+99. See Middleton+2018 he says the surface density can be obtained
    correcting for the mass loss of the disk
    R: astropy.quantity,
        The radius at which to estimate the density
    Mdot: astropy.quantity
        The mass-accretion rate
    Risco:
        Inner most stable orbit (see Shakura & Sunyaev 1973 after Equation 2.3)
    alpha: float
        Alpha viscosity
    """
    w = keplerian_angular_w(R, M)
    kT = (sigma_T / m_p).decompose(bases=[u.g, u.cm])
    factor = 64/9 * np.pi * c.to(u.cm/u.s)**2 / kT**2
    r = R.to(u.cm)
    Sigma = factor / alpha / w / (Mdot.to(u.g / u.s)) * (1 - (r / Risco.to(u.cm))**-0.5)
    return Sigma

def fastness_param(Rmag, Rco):
    """Computes the fastness parameter

    Parameters
    ----------
    Rm: astropy.quantity
        Magnetospheric radius
    Rco: astropy.quantity
        Co-rotation radius
    """
    return (Rmag.to(u.m) / Rco.to(u.m)) ** (3/2)


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
    omega = fastness_param(Rmag, Rco)
    N_0 = Mdot.to(u.g / u.s) * np.sqrt(G * M_NS * Rmag) # e.g. Ghosh & Lamb 1979 Eq 2
    return (xi * N_0 * (1 - omega)).decompose(bases=u.cgs.bases)


def matts_surface_density(Medd, Risco, Rin=None, M=1.4*M_sun, alpha=0.1):
    """Equation 11 from Middleton+2018. Note that in the Equaiton the radii should be in physical units, even if they are not given like that.
    Medd: astropy.quantity
        Eddington mass accretion rate
    Risco: astropy.quantity
        Inner most stable orbit
    Rin: astropy.quantity
        Inner disk radius (could be magnetospheric radius for a NS). If not given assume Risco
    """
    if Rin is None:
        Rin = Risco
    Sigma = 1 / alpha / 1000 * Medd.to(u.g/u.s)  / Risco.to(u.cm) * np.sqrt(Rin.to(u.cm) / G.decompose(bases=[u.s, u.g, u.cm]) / M.to(u.g))
    return Sigma

def magnetic_moment(B, R_NS=10*u.km):
    """See e.g. Equation (2) from Tsygankov

    """
    return B.to(u.G) * R_NS.to(u.cm)**3/2

def scale_height(m_r, R, Risco, Rs, efficiency):
    """Equation 18 from Lipunova+99, works for both sub and super critical disks without advection
        Just replace Mdot(R) by the appropiate calculation (i.e. with advection/outflows and so on)
    m_r:float
        Mass-transfer rate at every radii
    R: astropy.quantity
        Radii at which the scale height is to be calculated
    Risco: astropy.quantity
        Innermost stable orbit
    Rs: astropy.quantity
        Scharzschild radius (note that Lipunova calls it Rg)
    """
    H = Rs * m_r * 3 / 4 / efficiency * (1 - np.sqrt(Risco.to(u.cm)/R.to(u.cm)))
    return H

def scale_height_rad(M, Qrad, R):
    """Calculates the scale height using the balance between the gravitational force and the
    radiative energy Equation 17 in Lipunov+99"""
    H = Qrad * sigma_T.to(u.cm**2) / c.to(u.cm/u.s) * R.to(u.cm)**3 / m_p.to(u.g) / G.decompose(bases=[u.cm, u.g, u.s]) / M.to(u.g)
    return H

def ring_magnetic_precession(Rin, B, Sigma, H , R, beta = 30 * u.deg, theta=30 * u.deg, R_NS=10*u.km, M_NS=1.4 * u.M_sun):
    """Equation 2.35 from Lai+99. Returns the precession rate in angular velocity

    R: astropy.quantity
        The radius at which the precession is to be estimated
    B: astropy.quantity
        The magnetic field
    Sigma: astropy.quantity
        The disk surface density
    beta: astropy.quantity
        The angle between the disk normal axis and the NS spin axis
    theta: astropy.quantity
        The angle between the NS spin axis and the magnetic field
    """
    omega_w = keplerian_angular_w(R, M_NS)
    r = R.to(u.cm)
    mu = magnetic_moment(B, R_NS).to(u.G * u.cm**3).value * (u.cm**-0.5 * u.g**0.5 * u.s**-1 * u.cm**3) # conver Gauss into correct units
    D1 = np.sqrt(r**2 / Rin.to(u.cm)**2 - 1)
    D2 = np.sqrt(2 * H.to(u.cm) / Rin.to(u.cm))
    D = np.where(D1 < D2, D2, D1)
    prec = mu**2 / np.pi**2 / r**7 / omega_w / Sigma.to(u.g/u.cm**2) / D * np.cos(beta.to(u.rad)) * np.sin(theta.to(u.rad))**2
    return prec
