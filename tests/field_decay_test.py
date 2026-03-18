import unittest
from field_decay.field_decay_law import ZhangFieldDecayClassic, ZhangFieldDecayDiff
from field_decay.accretion import magnetospheric_radius, magnetic_moment
from field_decay.constants import M_suncgs


class TestZhangDecayConsistency(unittest.TestCase):
    """Test that the analytical (Classic) and differential Zhang decay laws agree
    when given the same constant mass-accretion rate.

    ZhangFieldDecayClassic (Eq 17 in Zhang & Kojima 2006) is the analytical
    solution to the ODE integrated step-by-step by ZhangFieldDecayDiff (Eq 16).
    For a constant Mdot, both should converge to the same B(M_acc) relation.
    """

    B_INIT = 1e14  # G — magnetar-like initial field
    M_NS = 1.4 * M_suncgs
    R_NS = 1e6  # cm
    MCRUST = 0.2 * M_suncgs
    XI = 0.1
    PSI = 0.5
    N_STEPS = 10_000  # fine enough for forward-Euler to agree within 0.5%
    TOTAL_MASS = 0.5 * 0.2 * M_suncgs  # 0.5 * Mcrust — enough decay to compare

    def _simulate(self, Mdot):
        """Evolve both laws with N equal-mass steps at constant Mdot.

        For the differential law the magnetospheric radius is recomputed at
        each step from the current B, matching the assumption of the
            analytical solution.

            Returns (B_classical, B_diff).
        """
        deltaM = self.TOTAL_MASS / self.N_STEPS

        classical = ZhangFieldDecayClassic(
            self.B_INIT,
            Mdot,
            self.M_NS,
            self.R_NS,
            Mcrust=self.MCRUST,
            xi=self.XI,
        )
        diff = ZhangFieldDecayDiff(
            self.B_INIT,
            self.R_NS,
            Mcrust=self.MCRUST,
            xi=self.XI,
        )

        for _ in range(self.N_STEPS):
            mu = magnetic_moment(diff.B, self.R_NS)
            rmag = magnetospheric_radius(Mdot, mu, self.M_NS, psi=self.PSI)
            diff.decay_field(deltaM, rmag)
            classical.decay_field(deltaM)

        return classical.B, diff.B

    def test_consistency_for_varying_mdot(self):
        """Differential and analytical laws must agree within 0.5% for a
        range of mass-accretion rates (increasing Mdot → stronger decay)."""
        mdot_values = [1e16, 1e17, 1e18, 1e19]  # g/s

        for Mdot in mdot_values:
            with self.subTest(Mdot=Mdot):
                B_classical, B_diff = self._simulate(Mdot)
                self.assertAlmostEqual(
                    B_diff / B_classical,
                    1.0,
                    places=2,
                    msg=(
                        f"Mdot={Mdot:.0e} g/s: Zhang differential "
                        f"(B={B_diff:.3e} G) and classical "
                        f"(B={B_classical:.3e} G) disagree by more than 0.5%"
                    ),
                )

    def test_field_decreases_with_higher_mdot(self):
        """A higher Mdot raises the bottom field Bf (Bf ∝ Mdot^½), so the
        field decays *less* and the final B is *higher* for higher Mdot.
        This monotonicity is verified for both laws independently."""
        mdot_values = [1e16, 1e17, 1e18, 1e19]
        B_classical_list = []
        B_diff_list = []

        for Mdot in mdot_values:
            B_cl, B_di = self._simulate(Mdot)
            B_classical_list.append(B_cl)
            B_diff_list.append(B_di)

        for i in range(len(mdot_values) - 1):
            self.assertLess(
                B_classical_list[i],
                B_classical_list[i + 1],
                msg=(
                    f"Classical: B should be less decayed (higher) at higher Mdot "
                    f"({mdot_values[i]:.0e} → {mdot_values[i+1]:.0e} g/s)"
                ),
            )
            self.assertLess(
                B_diff_list[i],
                B_diff_list[i + 1],
                msg=(
                    f"Differential: B should be less decayed (higher) at higher Mdot "
                    f"({mdot_values[i]:.0e} → {mdot_values[i+1]:.0e} g/s)"
                ),
            )


if __name__ == "__main__":
    unittest.main()
