#!/usr/bin/env python
# coding: utf-8
import numpy as np
from math import cos, sin, pi
import cython
from constants import M_suncgs, Gcgs, ccgs

# Enable Cython optimizations
if cython.compiled:
    from cython import boundscheck, wraparound
else:
    def boundscheck(x):
        return lambda f: f
    def wraparound(x):
        return lambda f: f



class CO():
    """Base compact object class"""
    
    # Cython attribute declarations
    M: cython.double
    Rg: cython.double
    spin: cython.double
    Risco: cython.double
    Medd: cython.double
    
    def __init__(self, M:  cython.double, spin:  cython.double=0):
        self.M = M
        self.Rg = gravitational_radius(self.M)
        self.spin = spin
        self.Risco = isco_radius(self.M, self.spin)
        efficiency: cython.double = accretion_efficiency(self.M, self.Risco)
        self.Medd = eddington_luminosity(self.M / M_suncgs)/ ccgs** 2 / efficiency
    
    def __str__(self):
        return (f"Compact Object (CO):\n"
            f"Mass: {self.M:.2e} g\n"
            f"Spin: {self.spin}\n"
            f"Risco: {self.Risco:.2e} cm\n"
            f"Eddington mass-accretion rate: {self.Medd:.2e} erg/s")



class NS(CO):
    """Neutron Star class"""
    
    # Cython attribute declarations
    R_NS: cython.double
    I: cython.double
    P_NS: cython.double
    omega: cython.double
    chi: cython.double
    alpha: cython.double
    eta: cython.float
    Rco: cython.double
    Rlc: cython.double
    
    def __init__(self, P_NS: cython.double, M_NS: cython.double=1.4 * M_suncgs, R_NS: cython.double=10**6, chi:  cython.double=np.pi / 4, alpha:  cython.double=np.pi / 4, eta: cython.float=0.1):
        """
        P_NS: float,
            Spin period in seconds
        R_NS: float,
            Radius of the NS in cm
        chi: float,
            Magnetic angle (measured from the spin axis; see e.g. Biryukov and Abolmasov 2021) in radiants
        alpha: float,
            Spin axis angle with respect to the disk (see e.g. Biryukov and Abolmasov 2021) in radiants
        eta: float,
            Coupling factor (0 < eta <= 1)
        """
        super().__init__(M_NS)
        self.R_NS = R_NS
        self.I = self.moment_of_inertia()
        self.update_period(P_NS) #  this is the setter
        self.chi = chi
        self.alpha = alpha
        if not (0 <= eta <= 1):
            raise ValueError("eta coupling factor needs to be 0 <= eta <=1 (%.1f)" % eta)
        self.eta = eta

    def __str__(self):
        return (f"Neutron Star (NS):\n"
            f"Mass: {self.M:.2e} g\n"
            f"Radius: {self.R_NS:.2e} cm\n"
            f"Spin period: {self.P_NS:.2f} s\n"
            f"Spin: {self.spin}\n"
            f"Omega (angular velocity): {self.omega:.2e} rad/s\n"
            f"Risco: {self.Risco:.2e} cm\n"
            f"Rco (corotation radius cm): {self.Rco:.2f}\n"
            f"Rlc (light cylinder cm): {self.Rlc:.2f}\n"
            f"Eddington mass-accretion rate: {self.Medd:.2e} g/s")

    @boundscheck(False)
    @wraparound(False)
    def torque(self, accretion_torque: cython.double, magnetic_torque: cython.double, braking_torque: cython.double, deltaT: cython.double) -> None:

        """Torque the NS, updating P_NS, chi and alpha based on the input torques. The NS parameters are updated internally
        
        Parameters
        ----------
        accretion_torque: float,
        magnetic_torque: float,
        braking_torque: float,
        deltaT: float,
            Span of time in seconds
        """
        
        sinchi: cython.double = sin(self.chi)
        coschi: cython.double = cos(self.chi)
        sincoschi: cython.double = sinchi * coschi
        sinalpha: cython.double = sin(self.alpha)
        cosalpha: cython.double = cos(self.alpha)
        Nspin: cython.double = accretion_torque * cosalpha + braking_torque * (1 + sinchi**2.) + magnetic_torque
        normfactor: cython.double = 1 - self.eta/2. * ( (sinchi * sinalpha)**2. + 2 * (coschi * cosalpha**2.)**2.)
        Nchi: cython.double = self.eta / normfactor * accretion_torque * sinalpha**2. * cosalpha * sincoschi + braking_torque * sincoschi
        Nalpha: cython.double = -accretion_torque * sinalpha

        omegaI: cython.double = self.omega * self.I 
        deltaTomegaI: cython.double  = deltaT / omegaI   
        deltaP: cython.double = -Nspin * self.P_NS  * deltaTomegaI
        deltachi: cython.double = Nchi * deltaTomegaI
        deltaalpha: cython.double = Nalpha * deltaTomegaI

        self.update_period(deltaP + self.P_NS)         
        self.chi += deltachi
        self.alpha += deltaalpha
    

    @boundscheck(False)
    @wraparound(False)
    def update_period(self, period: cython.double) -> None:
        """Set the period of the NS (in seconds)"""
        self.P_NS = period
        self.omega = 2. * pi / period # angular velocity
        self.spin = self.period_to_spin()
        self.Risco = isco_radius(self.M, self.spin)
        self.Rco = self.corotation_radius() # in units of ISCO
        self.Rlc = self.light_cylinder() # in units of ISCO
        efficiency: cython.double = accretion_efficiency(self.M, self.Risco)
        self.Medd = eddington_luminosity(self.M / M_suncgs)/ ccgs** 2. / efficiency
        

    def cos_function(self, ) -> cython.double:
        """Equation 20 combined with eta (see Eq 26) from Byryukov and Abolmasov 2021"""
        A: cython.double = 1 - self.eta/2. * (sin(self.chi)**2. * sin(self.alpha)**2. + 2 *cos(self.chi)**2. * cos(self.alpha)**2.)
        return self.eta / A


    def corotation_radius(self,)-> cython.double:
        """Return the corotation radius (e.g. equation (1) from Tysgankov et al 2016). Everything in cgs

        Parameters:
        -------
        P:float,
            Period of the NS in s
        M_NS:float
            Mass of the NS in g

        Returns the corotation radius in cm
        """
        return (Gcgs *  self.M / self.omega**2.) ** (1/3)

    def light_cylinder(self)-> cython.double:
        """Return the light cylinder (e.g. Biryukov and Abolmasov 2021)

        Parameters:
        -------
        P: float,
            Period of the NS in seconds

        Returns the light cylinder in cm
        """
        return ccgs / self.omega

    def moment_of_inertia(self,)-> cython.double:
        """Moment of Inertia of a NS (see http://hyperphysics.phy-astr.gsu.edu/hbase/isph.html). All units in cgs
        2 / 5 = 0.4
        Returns the moment of inertia in cgs
        """
        return (0.4 * self.M * self.R_NS**2.)
    

    def spin_to_period(self, a_spin:  cython.double)-> cython.double:
        """Convert spin of a NS to period
        
        Returns the period in seconds
        """
        P: cython.double = 2 * np.pi * self.I * ccgs / a_spin / Gcgs / self.M**2
        return P

    def period_to_spin(self)-> cython.double:
        """Convert Period to spin of the NS"""
        a_spin: cython.double = self.I * self.omega * ccgs / Gcgs / self.M**2
        return a_spin

#@jit(nopython=True)
@cython.cfunc
@boundscheck(False)
@wraparound(False)
def accretion_efficiency(M:  cython.double, R:  cython.double)-> cython.double:
    """Returns the accretion efficiency. Everything in cgs
    M: float
        Mass of the compact object in g
    R: float
        Radius of the compact object or innermost stable orbit in cm
    Returns the accretion efficiency
    """
    return Gcgs * M  / (2. * ccgs ** 2. * R)

#@jit(nopython=True)
def accretion_luminosity(M_dot: cython.double, M: cython.double=1.4 * M_suncgs, R: cython.double=10**6) -> cython.double:
    """Returns the accretion luminosity in erg/s (see Vasilopoulos et al 2019 paragraph after eq 8.
        M_dot: astropy.quantity,
            Mass-accretion rate in g/s
        M: float,
            Mass of the compact object in g
        R: float,
            Radius of the compact object or inner stable orbit in cm
        Returns the accretion luminosity in erg/s
    """
    efficiency: cython.double = accretion_efficiency(M, R)
    return efficiency * M_dot * ccgs ** 2.

#@jit(nopython=True)
@cython.cfunc
@boundscheck(False)
@wraparound(False)
def isco_radius(M:  cython.double, a:  cython.double=0.998)-> cython.double:
    """Returns the ISCO radius for a given mass in cm.
    Parameters
    ----------
    M: float,
        Mass of the compact object in cgs
    a: float,
        Dimensionless spin of the compact object: 0 for a Scharzschild black hole or 0.998 for a Kerr black hole.
    Returns the radius of the inner most stable orbit in cm
    """
    if a > 1:
        raise ValueError(f"Error in calculation of the ISCO radius. \nSpin parameter (a={a:.6f}) exceeds maximum allowed value of 1.")
    
    z1: cython.double = 1 + (1 - a**2.) ** (1/3) * ((1 + a)** (1/3) + (1-a) ** (1/3))
    z2: cython.double = np.sqrt(3. * a ** 2. + z1**2)
    # this implements the +- sign of a
    return (3. + z2 - a * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * gravitational_radius(M)

#@jit(nopython=True)
def schwarzschild_radius(M:  cython.double)-> cython.double:
    """Returns the scharzchild radius for a given mass in cm. 
        Parameters
        ----------
        M: float
            In grams
    """
    return 2. * gravitational_radius(M)

#@jit(nopython=True)
@cython.cfunc
@boundscheck(False)
@wraparound(False)
def gravitational_radius(M:  cython.double)-> cython.double:
    """Returns the gravitational radius for a given mass in g.
    Parameters
    ----------
    M: float
        Mass of the compact object in grams

    Returns the gravitational radius in cm
    """
    return (Gcgs * M/ ccgs**2.)

#@jit(nopython=True)
@cython.cfunc
@boundscheck(False)
@wraparound(False)
def eddington_luminosity(M:  cython.double)-> cython.double:
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: float
            Mass in solar units
        Returns the Eddington luminosity in erg/s (cgs)
    """
    return 1.26 * M * 10**38.

#@jit(nopython=True)
def eddington_accretion_rate(M: cython.double, R_in: cython.double) -> cython.double:
    """The classical Eddington luminosity for a given mass.
        Parameters
        ----------
        M: astropy.quantity
        R_in: astropy.quantity
        Returns the Eddington accretion rate in quantity
    """
    efficiency: cython.double = accretion_efficiency(M, R_in)
    # convert erg to cgs
    return eddington_luminosity(M) / efficiency / ccgs**2.


#@jit(nopython=True)
def magnetic_moment_to_B(mu: cython.double, Rns: cython.double = 10 **6) -> cython.double:
    """Compute magnetic field from magnetic moment. See after Equation (2) from Tsygankov et al 2016
        Returns the magnetic field in G units.
        Parameters
        ----------
        mu: float
            Magnetic moment in cgs units (i.e. Gauss * cm^3)
        
        Rns: float
            Radius of the neutron star in cm. Default 10^6 cm (1 km)"""
    B: cython.double = 2 * mu / (Rns) ** 3
    return B
