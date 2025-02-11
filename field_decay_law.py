import astropy.units as u
from astropy.constants import G
from math import sqrt, exp
import cython 

class BaseFieldDecayLaw:
    def __init__(self, B_init:  cython.double, name):
        """Parameters
        B_init: float,
            Initial magnetic field in Gauss
        """
        self.name = name
        self.B_init = B_init
        self.mass_accumulated:cython.double
        self.mass_accumulated = 0
        self.B = B_init

    def decay_field(self, deltaM:  cython.double, Rmag=None)-> None:
        """Decays the initial magnetic field based on an accreted mass
        
        deltaM: float,
            The mass accumulated in the next delta t (i.e. the new instantaneous mass accreted onto the NS Mdot x deltaT)
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __repr__(self) -> str:
        return self.name
    
class PayneFieldDecay(BaseFieldDecayLaw):
    def __init__(self, B_init:  cython.double, name="Payne", Mc=2*10**-4 * u.Msun, Mb=4.6*10**-5 * u.Msun):
        """Parameters
            Mc: astropy.units
            Mb: astropy.units
        """
        super().__init__(B_init, name)
           
        self.Mc = Mc.to(u.g).value
        self.Mb = Mb.to(u.g).value
        self.n = self.find_n()


    def find_n(self,):
        def func_n(n, K):
            return n**-2.25 - n**-3.25 -K

        n_a = 1.0001
        n_b = 3
        err_tol = 0.001
        err = (n_b - n_a) / 2
        K = (self.Mc/self.Mb)**-2.25
        f_na = func_n(n_a, K)

        n_c: cython.double
        f_nc: cython.double

        if K> 0.11:
            raise ValueError("K (%.2f) needs to be below 0.11, Mc needs to be greater than %.1e g/s!" % (K, self.Mc))

        while err > err_tol:
            n_c = (n_a + n_b) / 2
            f_nc = func_n(n_c, K)
            if f_na * f_nc <0:
                n_b = n_c
            else:
                n_a = n_c
                f_na = f_nc
            err = abs(n_b - n_a) / 2
        return n_c
    
    def decay_field(self, deltaM:  cython.double, Rmag=None)-> None:
        """Updates the B field after accreting Mdot over delta T.

        Equation 8 in Payne and Melatos 2007
        Equation 35 in Payne 2004
        """
        self.mass_accumulated+= deltaM
        if self.mass_accumulated < self.Mc / self.n:
            self.B = self.B_init *  (1 - self.mass_accumulated / self.Mc)
        else:
            self.B = self.B_init *  (self.mass_accumulated / self.Mb) ** (-2.25) # 9/4 = 2.25


class ZhangFieldDecayClassic(BaseFieldDecayLaw):
    """Magnetic field Decay law according to Zhang & Kojima 2006"""
    def __init__(self, B_init:  cython.double, Mdot:  cython.double, ns, name="Zhang", Mcrust=0.2 * u.M_sun, xi: cython.float=0.1):
        """Parameters
        
            ns:float,
                Neutron Star object
            Mcrust:astropy.units
                Mass of the NS crust
            xi: float,
                Parameter between zero and 1 which takes into account deviations from frozen-in plasma. Default 0.1

        """
        super().__init__(B_init, name)

        if not 0 <=xi <=1:
            raise ValueError("Parameter xi must be between 0 and 1!")

        self.Bf = self.bottom_field(ns, Mdot, xi=0.5)
        self.Mcrust = Mcrust.to(u.g).value
        self.xi = xi
        x0_2 = (self.Bf/self.B_init)**(4/7)
        self.C = 1 + sqrt(1 - x0_2)
        self.B = self.B_init

    
    def bottom_field(self, ns, Mdot: cython.double, psi:  cython.float=0.5):
        """Computes the bottom magnetic field according to Equation 18 

        Parameters
        ----------
        ns:float,
            Neutron Star object

        Returns the bottom magnetic field in G
        """
        Gcgs = G.to(u.cm**3/u.g/u.s**2).value
        #return 1.32 * 10**8 * (Mdot / ns.Medd) **(1/2) * ((ns.M*u.g).to(u.M_sun).value/1.4)**(1/4) * (ns.R_NS / 10**6)**(-5/4) * xi**(-7/4)
        return ((ns.R_NS/psi)**7 * 2 * Gcgs * ns.M * Mdot**2 / (ns.R_NS**12)) ** (1/4)
    
    def decay_field(self, deltaM:cython.double, Rmag=None):
        """Equation 17
        """
        self.mass_accumulated+= deltaM
        y = 2 * self.xi * self.mass_accumulated / (7 * self.Mcrust)
        self.B = self.Bf / (1 - (self.C / exp(y) - 1)**2)**(1.75) # 7/4
    
class ZhangFieldDecayDiff(BaseFieldDecayLaw):
    """Magnetic field Decay law according to Zhang & Kojima 2006"""
    def __init__(self, B_init:cython.double, R_NS:cython.float=10**6., Mcrust=0.2 * u.M_sun, xi:  cython.float=0.1, name="Zhang (Modified)"):
        """Parameters
        
            R_NS:float,
                Radius of the NS in cm
            Mcrust:astropy.units
                Mass of the NS crust
            xi: float,
                Parameter between zero and 1 which takes into account deviations from frozen-in plasma. Default 0.1

        """
        super().__init__(B_init, name)

        if not 0 <=xi <=1:
            raise ValueError("Parameter xi must be between 0 and 1!")

        self.Mcrust = Mcrust.to(u.g).value
        self.xi = xi
        self.B = B_init
        self.R_NS = R_NS
    
    def decay_field(self, deltaM:cython.double, Rmag:cython.double)-> None:
        """Equation 16
        """
        self.mass_accumulated+= deltaM
        rmag_rns = sqrt(Rmag - self.R_NS)
        deltaB = -self.xi * deltaM * rmag_rns * self.B / (self.Mcrust * (sqrt(Rmag) - rmag_rns))
        self.B = self.B + deltaB


class ShibazakiFieldDecay(BaseFieldDecayLaw):
    """Decay law according to et al. 1989"""
    def __init__(self, B_init:  cython.double, name="Shibazaki", Mb=1e-4 * u.M_sun):
        super().__init__(B_init, name)
        self.Mb = Mb.to(u.g).value


    def decay_field(self, deltaM:  cython.double, Rmag=None)-> None:
        """Returns the new B field
        
        deltaM: float,
            Mass accreted in g in a delta t
        """
        self.mass_accumulated+= deltaM
        self.B = self.B_init / (1. + self.mass_accumulated / self.Mb)