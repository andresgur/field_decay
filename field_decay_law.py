import astropy.units as u
from math import sqrt, exp

class BaseFieldDecayLaw:
    def __init__(self, B_init, name):
        """Parameters
        B_init: float,
            Initial magnetic field in Gauss
        """
        self.name = name
        self.B_init = B_init

    def decay_field(self, deltaM):
        """Decays the initial magnetic field based on an accreted mass"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
class PayneFieldDecay(BaseFieldDecayLaw):
    def __init__(self, B_init, name="Payne", Mc=2*10**-4 * u.Msun, Mb=4.6*10**-5 * u.Msun):
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

        if K> 0.11:
            raise ValueError("K (%.2f) needs to be below 0.11, Mc needs to be greater than %.1e %s!" % (K, Mc.value, Mc.unit))

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
    
    def decay_field(self, deltaM):
        """Returns the new B field after accreting Mdot over delta T.

        Equation 8 in Payne and Melatos 2007
        Equation 35 in Payne 2004

        Parameters
        ----------
        B: float
            Initial magnetic field
        deltaM: float,
            Mass accreted in g
        """
        if deltaM < self.Mc / self.n:
            return self.B_init *  (1 - deltaM / self.Mc)
        else:
            return self.B_init *  (deltaM / self.Mb) ** (-2.25) # 9/4 = 2.25


class ZhangFieldDecay(BaseFieldDecayLaw):
    """Magnetic field Decay law according to Zhang & Kojima 2006"""
    def __init__(self, B_init, name="Zhang", R_NS=10**6, Mcrust=0.2 * u.M_sun, xi=1):
        """Parameters
        
            R_NS:float,
                Radius of the NS in cm
            Mcrust:astropy.units
        """
        super().__init__(B_init, name)

        self.Bf = self.bottom_field(R_NS)
        self.Mcrust = Mcrust.to(u.g).value
        self.xi = xi
        x0_2 = (self.Bf/self.B_init)**(4/7)
        self.C = 1 + sqrt(1 - x0_2)

    
    def bottom_field(self, R_NS):
        """Computes the bottom magnetic field according to Equation 18 but assuming outflows within Rsph and assuming there's no advection
        The magnetospheric radius is taken from Middleton et al. 2023

        Parameters
        ----------
        R_NS:float,
            Radius of the NS in cm
        """
        return (R_NS / (4.2 * 10**7))**(9/4) * 10**12
    
    def decay_field(self, deltaM):
        """Equation 17
        """
        y = 2 * self.xi * deltaM / (7 * self.Mcrust)
        return self.Bf / (1 - (self.C / exp(y) - 1)**2)**(7/4)


class ShibazakiFieldDecay(BaseFieldDecayLaw):
    """Decay law according to et al. 1989"""
    def __init__(self, B_init, name="Shibazaki", Mb=1e-4 * u.M_sun):
        super().__init__(B_init, name)
        self.Mb = Mb.to(u.g).value


    def decay_field(self, deltaM):
        """Returns the new B field
        
        deltaM: float,
            Mass accreted in g
        """
        return self.B_init / (1 + deltaM / self.Mb)