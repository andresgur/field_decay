# co.py
from math import pi
import cython
from ..constants import M_suncgs, Gcgs, ccgs, m_pcgs, sigma_Tcgs


class CO:
    """Base compact object class."""

    M: cython.double
    Rg: cython.double
    spin: cython.double
    Risco: cython.double
    LEdd: cython.double
    MEdd: cython.double

    def __init__(self, M: cython.double, spin: cython.double = 0.0):
        self.M = M
        self.spin = spin

    @property
    def M(self) -> cython.double:
        return self._M

    @M.setter
    def M(self, value):
        # Assume solar masses on input
        self._M = float(value) * M_suncgs
        self._update_mass()

    @property
    def spin(self) -> cython.double:
        return self._spin

    @spin.setter
    def spin(self, value):
        if value > 1:
            raise ValueError(
                f"Spin parameter (a={value:.6f}) exceeds maximum allowed value of 1."
            )
        self._spin = value
        self._update_spin()

    def _update_mass(self):
        self.Rg = self.gravitational_radius()
        self.LEdd = self.eddington_luminosity()
        # MEdd depends on spin/efficiency, updated in _update_spin

    def _update_spin(self):
        self.Risco = self.isco_radius() * self.Rg
        eff = self.accretion_efficiency(self.Risco)
        self.MEdd = self.LEdd / (ccgs**2.0) / eff

    def __str__(self):
        return (
            "Compact Object (CO):\n"
            f"Mass: {self.M:.2e} g\n"
            f"Spin: {self.spin}\n"
            f"Risco: {self.Risco:.2e} cm\n"
            f"Eddington mass-accretion rate: {self.MEdd:.2e} g/s"
        )

    def gravitational_radius(self) -> cython.double:
        """Gravitational radius in cm."""
        return Gcgs * self.M / (ccgs**2.0)

    def isco_radius(self) -> cython.double:
        """ISCO radius in units of Rg."""
        z1 = 1 + (1 - self.spin**2.0) ** (1 / 3) * (
            (1 + self.spin) ** (1 / 3) + (1 - self.spin) ** (1 / 3)
        )
        z2 = (3.0 * self.spin**2.0 + z1**2) ** 0.5
        return 3.0 + z2 - self.spin * ((3 - z1) * (3 + z1 + 2 * z2)) ** 0.5

    def eddington_luminosity(self) -> cython.double:
        """Classical Eddington luminosity in erg/s (cgs)."""
        kappa = sigma_Tcgs / m_pcgs  # cm^2/g
        return 4.0 * pi * Gcgs * self.M * ccgs / kappa

    def eddington_accretion_rate(self, R_in: cython.double) -> cython.double:
        """Eddington accretion rate in g/s."""
        efficiency = self.accretion_efficiency(R_in)
        return self.LEdd / efficiency / (ccgs**2.0)

    def accretion_efficiency(self, R: cython.double) -> cython.double:
        """Accretion efficiency (dimensionless)."""
        return self.Rg / (2.0 * R)

    def Omega(self, R: float):
        """Calulate the Keplerian angular velocity at a given radius in cgs units

        Parameters
        ----------
        R: float or array-like,
            Radius at which to calculate the velocity in cm

        Returns
        -------
        Returns the Keplerian angular velocity in rad/s
        """
        return (Gcgs * self.M / R**3) ** 0.5
