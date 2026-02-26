# neutron_star.py
from math import pi, sin, cos
import cython
from ..constants import Gcgs, ccgs, M_suncgs
from ..field_decay_law import (
    ShibazakiFieldDecay,
    PayneFieldDecay,
    ZhangFieldDecayDiff,
)
from .co import CO

DECAY_LAWS = ["Payne", "Shibazaki", "Zhang"]


class NS(CO):
    """Neutron Star class."""

    R_NS: cython.double
    I: cython.double
    P: cython.double
    B: cython.double
    mu: cython.double
    omega: cython.double
    chi: cython.double
    alpha: cython.double
    eta: cython.float
    Rco: cython.double
    Rlc: cython.double

    def __init__(
        self,
        P: cython.double,
        M_NS: cython.double = 1.4,
        R_NS: cython.double = 10**6,
        B: cython.double = 10**12,
        chi: cython.double = pi / 4,
        alpha: cython.double = pi / 4,
        eta: cython.float = 0.1,
        decay_law: str = "Shibazaki",
    ):
        # Initialize base class with mass
        # we cannot call base class, cause we do not have the spin and the mass setter needs to update the moment of inertia too
        self.R_NS = R_NS
        self.M = M_NS
        self.chi = chi
        self.alpha = alpha
        if not (0 <= eta <= 1):
            raise ValueError(
                f"eta coupling factor needs to be 0 <= eta <= 1 ({eta:.2f})"
            )
        # inertia depends on M and R_NS
        self.eta = eta
        self.B = B  # sets mu
        self.P = P  # sets omega, Rco, Rlc, and syncs spin

        if decay_law in DECAY_LAWS:
            self.set_decay_law(decay_law)

    def __str__(self):
        return (
            "Neutron Star (NS):\n"
            f"Mass: {self.M / M_suncgs:.2e} M_sun\n"
            f"Radius: {self.R_NS / 1000.:.2e} km\n"
            f"Spin period: {self.P:.2f} s\n"
            f"Spin: {self.spin:.4f}\n"
            f"B (10**12 G) {self.B / 10**12:.2f}\n"
            f"Omega (angular velocity): {self.omega:.2e} rad/s\n"
            f"Risco: {self.Risco / 1000.:.2e} km\n"
            f"Rco (km): {self.Rco / 1000.:.2f}\n"
            f"Rlc (km): {self.Rlc / 1000.:.2f}\n"
            f"Eddington mass-accretion rate: {self.MEdd:.2e} g/s\n"
            f"Magnetic field decay law: {self.decay_law_implementation}"
        )

    # ------- Scalar properties -------
    @property
    def P(self) -> cython.double:
        return self._P

    @P.setter
    def P(self, value):
        if value <= 0:
            raise ValueError("Period must be positive!")
        self._P = value
        self._omega = 2.0 * pi / value  # angular velocity
        self.Rco = self.corotation_radius()
        self.Rlc = self.light_cylinder()
        # keep base-class spin consistent with period
        CO.spin.fset(self, self.period_to_spin())

    @property
    def M(self) -> cython.double:
        # Simply defer to the parent's implementation
        return self._M

    @M.setter
    def M(self, value):
        CO.M.fset(self, value)
        self.I = self.moment_of_inertia()

    @property
    def spin(self) -> cython.double:
        # Simply defer to the parent's implementation
        return self._spin

    @spin.setter
    def spin(self, value):
        # recalculate period
        self.P = self.spin_to_period(value)

    def set_decay_law(self, decay_law: str, **kwargs) -> None:
        """
        Switch the magnetic field decay law for the neutron star.

        Parameters
        ----------
        decay_law : str
            Name of the decay law to apply. Supported options:
            - "Zhang"      : Zhang & Kojima (diffusion-based) model.
            - "Shibazaki"  : Shibazaki et al. (1989) empirical model.
            - "Payne"      : Payne & Melatos (2004/2007) model.
        **kwargs : dict
            Additional keyword arguments passed to the selected decay law
            implementation. These allow customization of parameters such as
            crust mass, coupling factors, or geometry terms depending on the
            chosen model.

        Behavior
        --------
        - Uses the current magnetic field (`self.B`) as the initial field for
        the new decay law.
        - Replaces `self.decay_law_implementation` with an instance of the
        selected strategy class.
        - Raises ValueError if `decay_law` is not one of the supported options.

        Examples
        --------
        >>> ns.set_decay_law("Shibazaki")
        >>> ns.set_decay_law("Zhang", xi=0.1)
        """

        B_init = self.B
        if decay_law == "Zhang":
            self.decay_law_implementation = ZhangFieldDecayDiff(
                B_init, self.R_NS, **kwargs
            )
        elif decay_law == "Shibazaki":
            self.decay_law_implementation = ShibazakiFieldDecay(B_init, **kwargs)
        elif decay_law == "Payne":
            self.decay_law_implementation = PayneFieldDecay(B_init, **kwargs)
        else:
            raise ValueError(
                "Decay keyword %s not valid! Available options are: %s"
                % (decay_law, " ".join(DECAY_LAWS))
            )

    @property
    def omega(self) -> cython.double:
        return self._omega

    @property
    def B(self) -> cython.double:
        return self._B

    @B.setter
    def B(self, value):
        self._B = value
        self.mu = self.magnetic_moment()

    # ------- Physical relations -------
    def magnetic_moment(self):
        """μ = (B * R_NS^3) / 2 (cgs: Gauss*cm^3)."""
        return 0.5 * self.B * self.R_NS**3.0

    def corotation_radius(self) -> cython.double:
        """R_co = (G M / ω^2)^{1/3} (cgs)."""
        return (Gcgs * self.M / self.omega**2.0) ** (1.0 / 3.0)

    def light_cylinder(self) -> cython.double:
        """R_lc = c / ω (cgs)."""
        return ccgs / self.omega

    def moment_of_inertia(self) -> cython.double:
        """I ≈ 2/5 M R^2 (cgs)."""
        return 0.4 * self.M * self.R_NS**2.0

    # ------- Spin-period conversions -------
    def spin_to_period(self, a_spin: cython.double) -> cython.double:
        """Convert dimensionless spin to period (s)."""
        P = 2 * pi * self.I * ccgs / a_spin / Gcgs / (self.M**2)
        return P

    def period_to_spin(self) -> cython.double:
        """Convert period to dimensionless spin."""
        a_spin = self.I * self.omega * ccgs / Gcgs / (self.M**2)
        return a_spin

    # ------- Torques -------
    def braking_torque(self):
        """Simplified magneto-dipole braking torque: -μ^2 / R_lc^3."""
        return -(self.mu**2.0) / (self.Rlc**3.0)

    def decay_field(self, mass, Rmag=None):
        self.decay_law_implementation.decay_field(mass, Rmag)
        self.B = self.decay_law_implementation.B

    def cos_function(
        self,
    ) -> cython.double:
        """Equation 20 combined with eta (see Eq 26) from Byryukov and Abolmasov 2021"""
        A: cython.double = 1 - self.eta / 2.0 * (
            sin(self.chi) ** 2.0 * sin(self.alpha) ** 2.0
            + 2 * cos(self.chi) ** 2.0 * cos(self.alpha) ** 2.0
        )
        return self.eta / A

    def torque(
        self,
        accretion_torque: cython.double,
        magnetic_torque: cython.double,
        braking_torque: cython.double,
        deltaT: cython.double,
    ) -> None:
        """
        Update P, chi, alpha based on torques over a time step deltaT.
        """
        sinchi: cython.double = sin(self.chi)
        coschi: cython.double = cos(self.chi)
        sinalpha: cython.double = sin(self.alpha)
        cosalpha: cython.double = cos(self.alpha)

        Nspin: cython.double = (
            accretion_torque * cosalpha
            + braking_torque * (1 + sinchi**2.0)
            + magnetic_torque
        )

        normfactor: cython.double = 1 - self.eta / 2.0 * (
            (sinchi * sinalpha) ** 2.0 + 2 * (coschi * cosalpha) ** 2.0
        )
        # Eq 26 from Byryukov and Abolmasov 2021, with the sinchicoschi included in the normalization factor because it can be factored out
        Nchi: cython.double = (sinchi * coschi) * (
            (self.eta / normfactor) * accretion_torque * (sinalpha**2.0) * cosalpha
            + braking_torque
        )

        Nalpha: cython.double = -accretion_torque * sinalpha

        omegaI: cython.double = self.omega * self.I
        deltaTomegaI: cython.double = deltaT / omegaI

        deltaP: cython.double = -Nspin * self.P * deltaTomegaI
        deltachi: cython.double = Nchi * deltaTomegaI
        deltaalpha: cython.double = Nalpha * deltaTomegaI

        self.P = self.P + deltaP
        self.chi += deltachi
        self.alpha += deltaalpha
