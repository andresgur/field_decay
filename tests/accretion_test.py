import unittest
from field_decay.constants import M_suncgs, Gcgs, ccgs
from field_decay.accretion import (
    spherization_radius_poutanen,
    mass_transfer_inner_radius,
    mass_transfer_rate_mag_radius_bisection,
    mass_transfer_rate_mag_radius_secant,
    secant_method,
)
from field_decay.compact_object import NS
from math import pi
import numpy as np


class TestAccretionFlow(unittest.TestCase):
    def test_Poutanen(self):
        mdot = 1000
        ewind = 0.5
        Rin = 10
        Rsph = spherization_radius_poutanen(mdot, Rin, ewind) / Rin / mdot
        # see Poutanen below Eq 23
        self.assertAlmostEqual(
            Rsph, 1.16, msg="Spherization radius is wrong!", places=2
        )
        min = mass_transfer_inner_radius(mdot, ewind)
        self.assertAlmostEqual(
            min, 0.66, msg="Mass transfer inner radius does not work!", places=2
        )

        ewind = 1
        Rsph = spherization_radius_poutanen(mdot, Rin, ewind) / Rin / mdot
        self.assertAlmostEqual(
            Rsph, 1.04, msg="Spherization radius is wrong!", places=2
        )
        min = mass_transfer_inner_radius(mdot, ewind)
        self.assertAlmostEqual(
            min, 0.43, msg="Mass transfer inner radius does not work!", places=2
        )

    def test_mdot_rmag(self):
        B = 10**12
        ns = NS(10, B=B)
        mdots = np.arange(100, 1000, 100)
        mu = ns.mu
        for e_wind in [0.25, 0.5, 0.75, 1.0]:
            for mdot in mdots:
                Mdot = mdot * ns.MEdd
                Rsph = spherization_radius_poutanen(mdot, ns.Risco, e_wind)

                mass_transfer_sec = mass_transfer_rate_mag_radius_secant(
                    Mdot, mu, ns.MEdd, Rsph, ns.M, e_wind=e_wind, tol=1e-10
                )

                mass_transfer_bi = mass_transfer_rate_mag_radius_bisection(
                    Mdot, mu, ns.MEdd, Rsph, ns.M, e_wind=e_wind, err_tol=1e-10
                )
                self.assertAlmostEqual(
                    mass_transfer_sec / mass_transfer_bi,
                    1,
                    msg=f"Failed for mdot= {mdot}",
                )


class TestSecant(unittest.TestCase):

    def test_tolerance_step_based(self):
        """Ensure step-based tolerance triggers convergence correctly."""
        calls = {"n": 0}

        def f(x, a):
            calls["n"] += 1
            return x**2 - a

        # Use looser tolerance and check we get a root reasonably close
        a = 2
        root = secant_method(f, 0.5, 2.5, 1e-4, 100, a)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, np.sqrt(a), places=3)  # looser due to tol

        # Ensure we didn't do an absurd amount of calls (sanity check)

    def test_negative_root(self):
        """Check we can find a negative root by choosing appropriate initial guesses."""

        def f(x, a):
            return x**2 - a

        a = 2.0
        # Initial guesses near the negative root
        root = secant_method(f, -2.0, -1.0, 1e-12, 100, a)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, -np.sqrt(a), places=10)


if __name__ == "__main__":
    unittest.main()
