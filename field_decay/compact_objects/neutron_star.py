# neutron_star.py
from math import pi, sin, cos
import cython
from ..constants import Gcgs, ccgs, M_suncgs
from ..field_decay_law import (
    ShibazakiFieldDecay,
    PayneFieldDecay,
    ZhangFieldDecayDiff,
    ZhangFieldDecayClassic,
)
from .co import CO

DECAY_LAWS = ["Payne", "Shibazaki", "Zhang", "Zhang_simple"]
BRAKING_TORQUES = ["Dipole", "EnhancedDipole"]


class NS(CO):
    """Neutron Star class.

    Parameters
    ----------
    P: float,
        Spin period in seconds. The corresponding angular spin frequency is
        stored as omega = 2*pi/P and is used for the corotation and light
        cylinder radii.
    M_NS: float,
        Mass of the NS in grams
    R_NS: float,
        Radius of the NS in cm
    B: float,
        Magnetic field in Gauss
    chi: float,
        Angle between spin and magnetic axis in radians
    alpha: float,
        Angle between spin and accretion axis in radians
    eta: float,
        Coupling factor for the torque (0 <= eta <= 1)
    braking_torque: str,
        Braking torque prescription to use. Options: "Dipole", "EnhancedDipole"
    decay_law: str,
        Magnetic field decay law to use. Options: "Shibazaki", "Zhang
        (diff)", "Zhang (analytical)", "Payne"
    decay_law_kwargs: dict,
        Additional parameters for the decay law (e.g. xi for Zhang)
    """

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
        braking_torque: str = "Dipole",
        decay_law: str = "Shibazaki",
        **decay_law_kwargs,
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
        self.set_braking_torque(braking_torque)
        self.set_decay_law(decay_law, **decay_law_kwargs)

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
            f"Braking torque: {self.braking_torque_model}\n"
            f"Magnetic field decay law: {self.decay_law_implementation}"
        )

    # ------- Scalar properties -------
    @property
    def P(self) -> cython.double:
        """Spin period in seconds."""
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
    
    def set_braking_torque(self, braking_torque: str, **kwargs) -> None:
        """
        Switch the braking torque prescription for the neutron star.

        Parameters
        ----------
        braking_torque : str
            Name of the braking torque model to apply. Supported options:
            - "Dipole" : Standard magneto-dipole spin-down torque.
            - "EnhancedDipole" : Enhanced spin-down torque from disk-opened magnetic flux (Parfrey et al. 2016).

        Behavior
        --------
        - Replaces `self.braking_torque_model` with an instance of the selected strategy class.
        - Raises ValueError if `braking_torque` is not one of the supported options.

        Examples
        --------
        >>> ns.set_braking_torque("EnhancedDipole")
        """
        if braking_torque not in BRAKING_TORQUES:
            raise ValueError(
                "Braking torque keyword %s not valid! Available options are: %s"
                % (braking_torque, " ".join(BRAKING_TORQUES))
            )
        if braking_torque == "Dipole":
            self.braking_torque_model = DipoleBraking()
        elif braking_torque == "EnhancedDipole":
            self.braking_torque_model = EnhancedDipoleBraking(**kwargs)
        
    def set_decay_law(self, decay_law: str, **kwargs) -> None:
        """
        Switch the magnetic field decay law for the neutron star.

        Parameters
        ----------
        decay_law : str
            Name of the decay law to apply. Supported options:
            - "Zhang"      : Zhang & Kojima (diffusion-based) model (differential form).
            - "Shibazaki"  : Shibazaki et al. (1989) empirical model.
            - "Payne"      : Payne & Melatos (2004/2007) model.
            - "Zhang_simple" : Simplified Zhang & Kojima model (analytical form).
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
        if decay_law not in DECAY_LAWS:
            raise ValueError(
                "Decay keyword %s not valid! Available options are: %s"
                % (decay_law, " ".join(DECAY_LAWS))
            )
        B_init = self.B
        if decay_law == "Zhang":
            self.decay_law_implementation = ZhangFieldDecayDiff(
                B_init, self.R_NS, **kwargs
            )
        elif decay_law == "Zhang_simple":
            self.decay_law_implementation = ZhangFieldDecayClassic(
                B_init, self.R_NS, **kwargs
            )
        elif decay_law == "Shibazaki":
            self.decay_law_implementation = ShibazakiFieldDecay(B_init, **kwargs)
        elif decay_law == "Payne":
            self.decay_law_implementation = PayneFieldDecay(B_init, **kwargs)

    @property
    def omega(self) -> cython.double:
        """Angular spin frequency in rad/s. Defined as omega = 2 pi / P.
        """
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
    def braking_torque(self, Rmag: cython.double=None) -> cython.double:
        """Returns the strength of the braking torque according to the selected prescription."""
        return self.braking_torque_model.torque(self, Rmag)

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
        radial_twisting_torque: cython.double,
        braking_torque: cython.double,
        deltaT: cython.double,
    ) -> None:
        """
        Update P, chi, alpha based on torques over a time step deltaT.
        Parameters
        ----------
        accretion_torque: Torque from accretion (positive for spin-up).
        magnetic_torque: float,
            Torque from magnetically threaded disk (twisting of the Bz component; see Wang 1995;1997). This component is aligned with the disk axis
        radial_twisting_trque: float,
            Torque from radial shearing of the disk (see Wang 1997). This component is aligned with the disk axis
        braking_torque: float,
            Torque from magneto-dipole braking (negative for spin-down)
        deltaT: float,
            Time step over which to apply the torques (s)
        """
        sinchi: cython.double = sin(self.chi)
        sinchi2: cython.double = sinchi**2.0
        sinalpha: cython.double = sin(self.alpha)
        cosalpha: cython.double = cos(self.alpha)
        #  but it follows from the dependency of Bz with cos\chi and Bphi with Bz (see Wang 1997; Liu & Li 2021)
        # TODO: check where or how the radial_twisting torque this torque is aligned
        # note the magnetic torque and twisting torque already carry the cos/sin square chi factors
        Nspin: cython.double = (
            accretion_torque + magnetic_torque + radial_twisting_torque
        ) * cosalpha + braking_torque * (1 + sinchi2)

        # Eq 26 from Byryukov and Abolmasov 2021, with the sinchicoschi included in the normalization factor because it can be factored out
        # normfactor: cython.double = 1 - self.eta / 2.0 * (
        #    (sinchi * sinalpha) ** 2.0 + 2 * (coschi * cosalpha) ** 2.0
        # )
        # Nchi: cython.double = (sinchi * coschi) * (
        #    (self.eta / normfactor) * accretion_torque * (sinalpha**2.0) * cosalpha
        #    + braking_torque
        # )
        Nchi: cython.double = 0.0  # ignore chi evolution

        Nalpha: cython.double = -accretion_torque * sinalpha

        omegaI: cython.double = self.omega * self.I
        deltaTomegaI: cython.double = deltaT / omegaI

        deltaP: cython.double = -Nspin * self.P * deltaTomegaI
        deltachi: cython.double = Nchi * deltaTomegaI
        deltaalpha: cython.double = Nalpha * deltaTomegaI

        self.P = self.P + deltaP
        self.chi += deltachi
        self.alpha += deltaalpha


class BrakingTorqueModel:
    """Abstract interface for spin-down torque prescriptions.

    Implementations should return the torque applied to the neutron star.
    The current interface passes both the NS object and the instantaneous
    magnetospheric radius Rmag, which allows a torque model to depend on the
    disk truncation radius as well as the star's intrinsic properties.

    Parameters
    ----------
    zeta : float, optional
        Fixed opening efficiency / flux-opening factor for the braking-law
        model. This is stored once at construction time and reused by the
        torque implementation, rather than being revalidated on every call.
    """

    def __init__(self, zeta=1.0):
        self.zeta = zeta
        self._validate_zeta()

    def _validate_zeta(self):
        if not 0.0 <= self.zeta <= 1.0:
            raise ValueError(f"zeta must be between 0 and 1 (got {self.zeta:.2f})")
        
    def __repr__(self) -> str:
        return self.name

    def torque(self, Rmag=None):
        raise NotImplementedError


class DipoleBraking(BrakingTorqueModel):
    """Standard vacuum-dipole spin-down torque.

    This is the usual magneto-dipole estimate,

        N_dipole ~ -mu^2 / R_lc^3,

    and serves as the baseline torque. It depends on the stellar magnetic
    moment and spin frequency through the light cylinder, but not on Rmag.
    The model keeps its fixed zeta setting at construction time.
    """

    def __init__(self, zeta=1.0):
        super().__init__(zeta=zeta)
        self.name = "DipoleBraking"

    def torque(self, ns, Rmag=None):
        return -(ns.mu**2.0) / (ns.Rlc**3.0)


class EnhancedDipoleBraking(BrakingTorqueModel):
    """Enhanced spin-down torque from disk-opened magnetic flux.

    This model is motivated by Parfrey, Spitkovsky & Beloborodov (2016,
    ApJ 822, 33), who showed that differential rotation between a rapidly
    spinning neutron star and a surrounding accretion disk can open additional
    stellar magnetic flux. The resulting open-field wind can increase the
    spin-down torque above the standard dipole value.

    The implementation is intentionally written to accept the instantaneous
    magnetospheric radius Rmag, because the amount of opened flux and the
    effective wind torque are expected to depend on how far the disk is
    truncated from the star. In other words, this is an enhanced braking-law
    prescription for the disk–magnetosphere interaction, not a pure
    NS-parameter-only dipole formula.

    Use this model when you want the braking torque to reflect the stronger
    spin-down associated with disk-induced field opening and the open-flux
    wind mechanism described in the paper.

    Parameters
    ----------
    zeta : float, optional
        Fixed fraction of field lines that are opened by the disk. This is
        stored once at initialization and reused by the torque model.
    """

    def __init__(self, zeta=1.0):
        super().__init__(zeta=zeta)
        self.name = "EnhancedDipoleBraking"
    def torque(self, ns, Rmag):
        # The fixed opening factor is already stored in self.zeta.
        # Replace the placeholder below with the actual enhanced prescription
        # once the Rmag-dependent formula is finalized.
        return -self.zeta**2 * (ns.mu**2.0) / (Rmag**2.0) * ns.omega / ccgs